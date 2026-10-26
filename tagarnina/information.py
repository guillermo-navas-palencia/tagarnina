"""Information shared by the tagarnina algorithms."""

from tagarnina import __version__


def banner_lines(subtitle: str, width: int) -> list[str]:
    rule = "-" * width
    return [
        rule,
        f"tagarnina v{__version__}".center(width),
        subtitle.center(width),
        "(c) Guillermo Navas-Palencia, 2026, Apache License 2.0".center(width),
        rule,
    ]
