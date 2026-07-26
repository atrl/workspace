from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .stage1 import build_stage1_dashboard_payload, resolve_stage1_dashboard_source


APP_HTML = Path(__file__).with_name("stage1.html").read_text(encoding="utf-8")


def create_stage1_app(root: Path | str | None = None) -> FastAPI:
    source = resolve_stage1_dashboard_source(root)
    app = FastAPI(title="Stage1 Qlib Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return APP_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "source": "qlib_recorder", "recorder_id": source.recorder_id}

    @app.get("/api/stage1")
    def stage1(
        start: str | None = Query(default=None),
        end: str | None = Query(default=None),
    ) -> dict:
        try:
            return build_stage1_dashboard_payload(root=source.root, start=start, end=end)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage1 qlib recorder dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_stage1_app(args.root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
