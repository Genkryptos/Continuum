from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContentFact:
    text: str
    topic: str
    fact_key: str
    as_of: dt.date | None = None
    session_ids: list[str] = field(default_factory=list)


def route_topic(fact: ContentFact) -> str:
    text = f"{fact.topic} {fact.fact_key} {fact.text}".lower()
    if any(
        term in text
        for term in ("spotify", "playlist", "music", "lo-fi", "ambient")
    ):
        return "music"
    if "commute" in text:
        return "commute"
    if any(term in text for term in ("target", "coupon", "cartwheel", "shopping")):
        return "shopping"
    if any(term in text for term in ("degree", "university", "school", "education")):
        return "education"
    if any(term in text for term in ("fitness", "run", "workout", "health")):
        return "health_fitness"
    if any(term in text for term in ("guitar", "hobby", "theater", "play")):
        return "hobbies"
    return "misc"


_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")
_SPOTIFY_PLAYLIST_RE = re.compile(
    r"playlist on Spotify that I created,\s*called\s+([^,.]+)",
    re.IGNORECASE,
)
_COMMUTE_RE = re.compile(
    r"commute,\s*which takes\s+([^,.]+)",
    re.IGNORECASE,
)
_COMMUTE_DIRECT_RE = re.compile(
    r"(?:my|daily|the user's)?\s*daily commute takes\s+([^,.]+)",
    re.IGNORECASE,
)
_COUPON_RE = re.compile(
    r"redeemed a\s+\$?(\d+)\s+coupon on\s+([^,.]+)",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(r"degree in\s+([^,.]+)", re.IGNORECASE)
_PLAY_RE = re.compile(
    r"(?:production of|play called|play named)\s+([^,.?]+)",
    re.IGNORECASE,
)


def parse_eval_date(date_text: str | None) -> dt.date | None:
    if not date_text:
        return None
    try:
        return dt.datetime.fromisoformat(date_text).date()
    except ValueError:
        pass
    match = _DATE_RE.search(date_text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return dt.date(year, month, day)


def extract_content_facts(
    *,
    content: str,
    role: str,
    session_id: str,
    date_text: str | None = None,
    session_hint: str | None = None,
) -> list[ContentFact]:
    if role != "user":
        return []

    as_of = parse_eval_date(date_text)
    facts: list[ContentFact] = []

    spotify = _SPOTIFY_PLAYLIST_RE.search(content)
    if spotify:
        playlist = spotify.group(1).strip().strip('"')
        facts.append(
            ContentFact(
                text=f'The user created a Spotify playlist called "{playlist}".',
                topic="music",
                fact_key="spotify_playlist",
                as_of=as_of,
                session_ids=[session_id],
            )
        )

    commute = _COMMUTE_RE.search(content) or _COMMUTE_DIRECT_RE.search(content)
    if commute:
        duration = commute.group(1).strip()
        facts.append(
            ContentFact(
                text=f"The user's daily commute takes {duration}.",
                topic="commute",
                fact_key="commute_duration",
                as_of=as_of,
                session_ids=[session_id],
            )
        )

    coupon = _COUPON_RE.search(content)
    if coupon and session_hint and session_hint.lower() == "target":
        amount, product = coupon.groups()
        product_clean = re.sub(
            r"\s+(?:last|this|next)\s+\w+$", "", product.strip(),
            flags=re.IGNORECASE,
        )
        facts.append(
            ContentFact(
                text=f"The user redeemed a ${amount} {product_clean} coupon at Target.",
                topic="shopping",
                fact_key="target_coupon_redeemed",
                as_of=as_of,
                session_ids=[session_id],
            )
        )

    degree = _DEGREE_RE.search(content)
    if degree:
        subject = degree.group(1).strip()
        facts.append(
            ContentFact(
                text=f"The user graduated with a degree in {subject}.",
                topic="education",
                fact_key="degree",
                as_of=as_of,
                session_ids=[session_id],
            )
        )

    play = _PLAY_RE.search(content)
    if play and "play" in content.lower():
        title = play.group(1).strip().strip('"')
        facts.append(
            ContentFact(
                text=(
                    f'The user attended the play "{title}" at the local '
                    "community theater."
                ),
                topic="hobbies",
                fact_key="theater_play",
                as_of=as_of,
                session_ids=[session_id],
            )
        )

    return facts


def _format_fact(fact: ContentFact) -> str:
    if fact.as_of:
        return f"- As of {fact.as_of.isoformat()}, {fact.text[0].lower()}{fact.text[1:]}"
    return f"- {fact.text}"


def build_content_wiki_pages(facts: list[ContentFact]) -> dict[str, str]:
    by_topic: dict[str, list[ContentFact]] = defaultdict(list)
    for fact in facts:
        topic = fact.topic or route_topic(fact)
        by_topic[topic].append(fact)

    pages: dict[str, str] = {}
    for topic, topic_facts in by_topic.items():
        title = topic.replace("_", " ").title()
        grouped: dict[str, list[ContentFact]] = defaultdict(list)
        for fact in topic_facts:
            grouped[fact.fact_key].append(fact)

        current: list[ContentFact] = []
        older: list[ContentFact] = []
        for key_facts in grouped.values():
            ordered = sorted(
                key_facts,
                key=lambda fact: fact.as_of or dt.date.min,
                reverse=True,
            )
            current.append(ordered[0])
            older.extend(ordered[1:])

        lines = [f"# {title}", "", "## Current Facts", ""]
        for fact in sorted(current, key=lambda fact: fact.fact_key):
            lines.append(_format_fact(fact))
        if older:
            lines.extend(["", "## Older Facts", ""])
            for fact in sorted(
                older,
                key=lambda fact: (fact.fact_key, fact.as_of or dt.date.min),
                reverse=True,
            ):
                lines.append(_format_fact(fact))
        pages[f"{topic}.md"] = "\n".join(lines)

    return pages
