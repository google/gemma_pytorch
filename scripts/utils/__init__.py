def strtobool(val: str) -> bool:
    if val in {"1", "true"}:
        return True
    return False
