import pytest
from qiskit.providers.fake_provider import FakeBackend


@pytest.fixture
def mock_backend():
    """Provide a mock backend for testing."""
    return FakeBackend()


@pytest.fixture
def mock_sampler():
    """Provide a mock sampler for testing."""

    class MockSampler:
        def run(self, circuits, **kwargs):
            class MockResult:
                def result(self):
                    return {"00": 0.5, "11": 0.5}

            return MockResult()

    return MockSampler()
