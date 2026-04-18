# Expectations

- Validation peak may occur before `max_iters`
- Must not be treated as production-candidate purely because staged selection improves validation
- Success criteria are diagnostic:
  - identify best checkpoint iteration
  - show whether checkpoint choice changes fixed-split gate outcome materially
