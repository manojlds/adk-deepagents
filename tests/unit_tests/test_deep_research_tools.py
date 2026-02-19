from __future__ import annotations

from examples.deep_research import tools


def test_resolve_auto_provider_serper_first(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "y")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "z")
    assert tools._resolve_auto_provider() == "serper"


def test_resolve_auto_provider_falls_back_to_duckduckgo(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert tools._resolve_auto_provider() == "duckduckgo"


def test_web_search_auto_uses_serper_when_key_present(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_SEARCH_PROVIDER", "auto")
    monkeypatch.setenv("SERPER_API_KEY", "x")

    called: list[str] = []

    def fake_dispatch(provider: str, query: str, max_results: int, topic: str):
        called.append(provider)
        return [{"title": "A", "url": "https://a", "content": "x", "provider": provider}]

    monkeypatch.setattr(tools, "_dispatch_provider", fake_dispatch)
    result = tools.web_search("test")

    assert called == ["serper"]
    assert "Provider: serper" in result


def test_web_search_hard_fails_on_selected_provider_error(monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_SEARCH_PROVIDER", "serper")

    def fail_dispatch(provider: str, query: str, max_results: int, topic: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "_dispatch_provider", fail_dispatch)
    result = tools.web_search("test")
    assert "Web search error via provider 'serper'" in result
