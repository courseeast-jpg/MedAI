"""Deterministic Phase 1 execution layer."""

from execution.jobs import ExecutionJob, ExecutionResult

__all__ = ["ExecutionJob", "ExecutionResult", "ExecutionPipeline"]


def __getattr__(name: str):
    if name == "ExecutionPipeline":
        from execution.pipeline import ExecutionPipeline

        return ExecutionPipeline
    raise AttributeError(name)
