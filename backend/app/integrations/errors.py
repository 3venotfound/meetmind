from enum import Enum
from typing import Literal


IntegrationComponent = Literal["ai", "cv"]


class IntegrationErrorCode(str, Enum):
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    EXECUTION_FAILED = "execution_failed"
    INVALID_RESULT = "invalid_result"
    UNSAFE_PATH = "unsafe_path"


class IntegrationError(RuntimeError):
    def __init__(
        self,
        component: IntegrationComponent,
        code: IntegrationErrorCode,
        public_message: str,
        *,
        retryable: bool,
    ):
        super().__init__(public_message)
        self.component = component
        self.code = code
        self.public_message = public_message
        self.retryable = retryable


def not_configured(component: IntegrationComponent) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.NOT_CONFIGURED,
        f"{component.upper()} integration is not configured",
        retryable=False,
    )


def unavailable(component: IntegrationComponent) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.UNAVAILABLE,
        f"{component.upper()} integration is unavailable",
        retryable=False,
    )


def timed_out(component: IntegrationComponent) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.TIMEOUT,
        f"{component.upper()} processing timed out",
        retryable=True,
    )


def execution_failed(component: IntegrationComponent) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.EXECUTION_FAILED,
        f"{component.upper()} processing failed",
        retryable=True,
    )


def invalid_result(component: IntegrationComponent) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.INVALID_RESULT,
        f"{component.upper()} returned an invalid result",
        retryable=False,
    )


def unsafe_path(component: IntegrationComponent) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.UNSAFE_PATH,
        f"{component.upper()} returned an unsafe path",
        retryable=False,
    )
