"""Channel registry: every source a symbol's news can come from, built from configs/news.yaml."""

from __future__ import annotations

from news.channels.agent_reach import AgentReachChannel
from news.channels.base import Channel, ChannelStatus
from news.channels.public import (
    CseAnnouncementsChannel,
    GdeltChannel,
    GoogleNewsChannel,
    JinaWebChannel,
    NewsApiChannel,
    RssChannel,
    YouTubeChannel,
)


def build_channels(config: dict | None = None) -> list[Channel]:
    """All channels, available or not. Availability is checked per run via status()."""
    if config is None:
        from news.keywords import load_config

        config = load_config()
    channels: list[Channel] = [
        CseAnnouncementsChannel(),
        GoogleNewsChannel(),
        GdeltChannel(),
        RssChannel(config.get("rss_feeds") or []),
        NewsApiChannel(),
        JinaWebChannel(),
        YouTubeChannel(),
    ]
    fields = config.get("agent_reach_fields") or {}
    for name, spec in (config.get("agent_reach") or {}).items():
        channels.append(AgentReachChannel(name, spec, fields))
    return channels


def channel_statuses() -> list[ChannelStatus]:
    return [channel.status() for channel in build_channels()]


__all__ = ["Channel", "ChannelStatus", "build_channels", "channel_statuses"]
