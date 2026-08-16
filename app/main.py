import logging
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from db import get_engine
from rate_limit import enforce_rate_limit
from security import require_api_key
from schema_cache import get_schema as get_cached_schema
from schema_filter import filter_relevant_tables
from prompt_constructor import build_prompt
from llm_client import get_llm_client
from guardrail import apply_guardrails
from query_executor import execute_readonly
from back_translation import back_translate, score_alignment
from sanity_checker import check_result_sanity
from multi_query_validator import build_alternative_prompt, compare_results
from confidence_scorer import compute_confidence, compute_schema_coverage
from metrics import metrics_store
from logging_config import configure_logging

configure_logging()
logger = logging.getLogger("api")

app = FastAPI(title="Text-to-SQL Guardrails API")


class QueryRequest(BaseModel):
    question: str


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    """Serve the single-page UI from the same origin as the API.

    Same-origin is the point: the browser calls /v1/query directly with no CORS
    layer, and no API key is ever baked into the served HTML — the page asks the
    user for one and keeps it in localStorage.
    """
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    """Unauthenticated on purpose — platform health checks can't send a key,
    and this touches neither the database nor the LLM."""
    return {"status": "ok"}


@app.get("/schema", dependencies=[Depends(require_api_key)])
def get_schema():
    return get_cached_schema(get_engine())


@app.post(
    "/v1/query",
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
def run_query(req: QueryRequest):
    engine = get_engine()
    llm_client = get_llm_client()

    tables = get_cached_schema(engine)
    relevant_tables = filter_relevant_tables(req.question, tables)
    expected_table_names = [t.table.name for t in relevant_tables]
    prompt = build_prompt(req.question, relevant_tables)

    # ------------------------------------------------------------------
    # Generate SQL
    # ------------------------------------------------------------------

    generation = llm_client.generate_sql(prompt)
    guardrail_result = apply_guardrails(generation.sql)

    if not guardrail_result.allowed:
        metrics_store.record_query(
            blocked=True,
            confidence=None,
            had_flags=False,
        )

        return {
            "question": req.question,
            "generated_sql": generation.sql,
            "executed": False,
            "blocked_reason": guardrail_result.reason,
        }

    # ------------------------------------------------------------------
    # Execute SQL
    # ------------------------------------------------------------------

    execution = execute_readonly(engine, guardrail_result.sql)

    if execution.error:
        metrics_store.record_query(
            blocked=False,
            confidence=None,
            had_flags=True,
        )

        return {
            "question": req.question,
            "generated_sql": guardrail_result.sql,
            "executed": False,
            "error": execution.error,
        }

    # ------------------------------------------------------------------
    # Hallucination Detection
    # ------------------------------------------------------------------

    back_translated_question = back_translate(
        guardrail_result.sql,
        llm_client,
    )

    alignment_score = score_alignment(
        req.question,
        back_translated_question,
    )

    sanity_result = check_result_sanity(
        execution.rows,
    )

    alt_prompt = build_alternative_prompt(prompt)
    alt_generation = llm_client.generate_sql(alt_prompt)
    alt_guardrail_result = apply_guardrails(alt_generation.sql)

    if alt_guardrail_result.allowed:
        alt_execution = execute_readonly(
            engine,
            alt_guardrail_result.sql,
        )

        agreement_score = (
            compare_results(
                execution.rows,
                alt_execution.rows,
            )
            if not alt_execution.error
            else 0.0
        )
    else:
        agreement_score = 0.0

    schema_coverage = compute_schema_coverage(
        generation.tables_used,
        expected_table_names,
    )

    confidence = compute_confidence(
        syntax_valid=True,
        back_translation_alignment=alignment_score,
        sanity_passed=sanity_result.passed,
        sanity_issues=sanity_result.issues,
        multi_query_agreement=agreement_score,
        schema_coverage=schema_coverage,
    )

    logger.info(
        "query_complete",
        extra={
            "event": "query_complete",
            "question": req.question,
            "confidence": confidence.final_score,
            "flagged_issues": confidence.flagged_issues,
            "row_count": execution.row_count,
            "execution_time_ms": execution.execution_time_ms,
        },
    )

    metrics_store.record_query(
        blocked=False,
        confidence=confidence.final_score,
        had_flags=len(confidence.flagged_issues) > 0,
    )

    return {
        "question": req.question,
        "generated_sql": guardrail_result.sql,
        "explanation": generation.explanation,
        "executed": True,
        "rows": execution.rows,
        "row_count": execution.row_count,
        "execution_time_ms": execution.execution_time_ms,
        "confidence": confidence.final_score,
        "confidence_breakdown": confidence.signals,
        "flagged_issues": confidence.flagged_issues,
        "back_translated_question": back_translated_question,
    }


@app.get("/v1/metrics", dependencies=[Depends(require_api_key)])
def get_metrics():
    # These counters live in this instance's memory. On a serverless platform
    # that means they cover one instance's lifetime, not the deployment — two
    # consecutive calls can hit different instances and report different
    # numbers. Persist to a table if you need real aggregates.
    return {**metrics_store.summary(), "scope": "instance"}