#!/usr/bin/env python3
import json
import os
import sys
import urllib.request


LINEAR_API_URL = "https://api.linear.app/graphql"
TEAM_KEY = "SOX"
PROJECT_NAME = "SOXL Harness"
TODO_STATE_NAME = "Todo"


def graphql(query: str, variables: dict | None = None) -> dict:
    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        raise SystemExit("LINEAR_API_KEY is required")
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        LINEAR_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())
    if "errors" in data:
        raise SystemExit(json.dumps(data["errors"], ensure_ascii=False, indent=2))
    return data["data"]


def resolve_ids() -> tuple[str, str, str]:
    data = graphql(
        """
        query ResolveProjectAndStates($teamKey: String!) {
          teams(first: 20, filter: { key: { eq: $teamKey } }) {
            nodes {
              id
              key
              name
              states {
                nodes {
                  id
                  name
                }
              }
            }
          }
          projects(first: 50) {
            nodes {
              id
              name
            }
          }
        }
        """,
        {"teamKey": TEAM_KEY},
    )
    teams = data["teams"]["nodes"]
    if not teams:
        raise SystemExit(f"Could not find team with key {TEAM_KEY}")
    team = teams[0]
    todo_states = [s for s in team["states"]["nodes"] if s["name"] == TODO_STATE_NAME]
    if not todo_states:
        raise SystemExit(f"Could not find state named {TODO_STATE_NAME}")
    projects = [p for p in data["projects"]["nodes"] if p["name"] == PROJECT_NAME]
    if not projects:
        raise SystemExit(f"Could not find project named {PROJECT_NAME}")
    return team["id"], projects[0]["id"], todo_states[0]["id"]


def create_issue(title: str, description: str) -> dict:
    team_id, project_id, state_id = resolve_ids()
    return graphql(
        """
        mutation CreateIssue($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue {
              id
              identifier
              title
              url
            }
          }
        }
        """,
        {
            "input": {
                "teamId": team_id,
                "projectId": project_id,
                "stateId": state_id,
                "title": title,
                "description": description,
            }
        },
    )["issueCreate"]["issue"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: create_linear_soxl_issue.py <title> [description_file]"
        )
    title = sys.argv[1]
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            description = f.read()
    else:
        description = (
            "Goal:\\n"
            "- improve the SOXL harness with one documented loop\\n\\n"
            "Constraints:\\n"
            "- SOXL only\\n"
            "- keep non-leaking evaluation\\n"
            "- write proof of work back to the repo\\n\\n"
            "Reference files:\\n"
            "- AGENTS.md\\n"
            "- SPEC.md\\n"
            "- codex_exec_plans/active/2026-04-10-soxl-agent-loop.md\\n"
        )
    issue = create_issue(title, description)
    print(json.dumps(issue, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
