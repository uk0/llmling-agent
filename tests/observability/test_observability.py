"""Tests for the simplified observability system using Logfire."""

import os
from unittest.mock import patch

from pydantic import SecretStr
import pytest

from llmling_agent.observability.observability_registry import ObservabilityRegistry
from llmling_agent_config.observability import (
    CustomObservabilityConfig,
    LangsmithObservabilityConfig,
    LogfireObservabilityConfig,
    ObservabilityConfig,
)


@pytest.fixture
def clean_env():
    """Clean OTEL environment variables before and after tests."""
    env_vars = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_RESOURCE_ATTRIBUTES",
    ]

    # Store original values
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


def test_logfire_config_post_init():
    """Test that Logfire config computes private attributes correctly."""
    # Test US region
    config = LogfireObservabilityConfig(
        token=SecretStr("test_token_123"), region="us", service_name="test-service"
    )

    assert config._endpoint == "https://logfire-us.pydantic.dev"
    assert config._headers == {"Authorization": "Bearer test_token_123"}

    # Test EU region
    eu_config = LogfireObservabilityConfig(token=SecretStr("test_token_456"), region="eu")

    assert eu_config._endpoint == "https://logfire-eu.pydantic.dev"
    assert eu_config._headers == {"Authorization": "Bearer test_token_456"}


def test_langsmith_config_post_init():
    """Test that Langsmith config computes private attributes correctly."""
    config = LangsmithObservabilityConfig(
        api_key=SecretStr("ls_test_key"), project_name="test-project"
    )

    assert config._endpoint == "https://api.smith.langchain.com"
    assert config._headers == {"x-api-key": "ls_test_key"}


def test_custom_config():
    """Test custom observability config."""
    config = CustomObservabilityConfig(
        endpoint="https://my-custom-endpoint.com",
        headers={"Authorization": "Bearer custom_token"},
        service_name="my-service",
    )

    assert config.endpoint == "https://my-custom-endpoint.com"
    assert config.headers == {"Authorization": "Bearer custom_token"}


@patch("logfire.configure")
def test_registry_configure_logfire(mock_configure, clean_env):
    """Test registry configuration with Logfire backend."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(
        enabled=True,
        provider=LogfireObservabilityConfig(
            token=SecretStr("test_token"), service_name="test-service", environment="test"
        ),
    )

    registry.configure_observability(config)

    # Check that OTEL environment variables were set
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://logfire-us.pydantic.dev"
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert "Authorization=Bearer test_token" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "service.name=test-service" in os.environ["OTEL_RESOURCE_ATTRIBUTES"]
    assert "deployment.environment.name=test" in os.environ["OTEL_RESOURCE_ATTRIBUTES"]

    # Check that Logfire was configured
    mock_configure.assert_called_once_with(
        service_name="test-service", environment="test", send_to_logfire=True
    )


@patch("logfire.configure")
def test_registry_configure_langsmith(mock_configure, clean_env):
    """Test registry configuration with Langsmith backend."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(
        enabled=True,
        provider=LangsmithObservabilityConfig(
            api_key=SecretStr("ls_key"),
            project_name="test-project",
            service_name="my-service",
        ),
    )

    registry.configure_observability(config)

    # Check environment variables
    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://api.smith.langchain.com"
    assert "x-api-key=ls_key" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    # Check that Logfire was configured to NOT send to logfire
    mock_configure.assert_called_once_with(
        service_name="my-service", environment=None, send_to_logfire=False
    )


@patch("logfire.configure")
def test_registry_configure_custom(mock_configure, clean_env):
    """Test registry configuration with custom backend."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(
        enabled=True,
        provider=CustomObservabilityConfig(
            endpoint="https://my-otel-collector.com:4318",
            headers={"X-API-Key": "custom_key"},
            service_name="custom-service",
            environment="prod",
        ),
    )

    registry.configure_observability(config)

    # Check environment variables
    assert (
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "https://my-otel-collector.com:4318"
    )
    assert "X-API-Key=custom_key" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]

    mock_configure.assert_called_once_with(
        service_name="custom-service", environment="prod", send_to_logfire=False
    )


@patch("logfire.configure")
def test_registry_disabled(mock_configure, clean_env):
    """Test that disabled observability doesn't configure anything."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(enabled=False)
    registry.configure_observability(config)

    mock_configure.assert_not_called()
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ


@patch("logfire.configure")
def test_registry_no_provider(mock_configure, clean_env):
    """Test that no provider doesn't configure anything."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(enabled=True, provider=None)
    registry.configure_observability(config)

    mock_configure.assert_not_called()


@patch("logfire.configure")
def test_registry_double_configuration(mock_configure, clean_env):
    """Test that registry prevents double configuration."""
    registry = ObservabilityRegistry()

    config = ObservabilityConfig(
        enabled=True, provider=LogfireObservabilityConfig(token=SecretStr("test"))
    )

    # Configure twice
    registry.configure_observability(config)
    registry.configure_observability(config)

    # Should only be called once
    assert mock_configure.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
