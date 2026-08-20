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
        diagnostic: object | None = None,
    ):
        super().__init__(public_message)
        self.component = component
        self.code = code
        self.public_message = public_message
        self.retryable = retryable
        self.diagnostic = diagnostic


def not_configured(
    component: IntegrationComponent,
    *,
    diagnostic: object | None = None,
) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.NOT_CONFIGURED,
        f"{component.upper()} integration is not configured",
        retryable=False,
        diagnostic=diagnostic,
    )


def unavailable(
    component: IntegrationComponent,
    *,
    diagnostic: object | None = None,
) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.UNAVAILABLE,
        f"{component.upper()} integration is unavailable",
        retryable=False,
        diagnostic=diagnostic,
    )


def timed_out(
    component: IntegrationComponent,
    *,
    diagnostic: object | None = None,
) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.TIMEOUT,
        f"{component.upper()} processing timed out",
        retryable=True,
        diagnostic=diagnostic,
    )


def execution_failed(
    component: IntegrationComponent,
    *,
    diagnostic: object | None = None,
) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.EXECUTION_FAILED,
        f"{component.upper()} processing failed",
        retryable=True,
        diagnostic=diagnostic,
    )


def invalid_result(
    component: IntegrationComponent,
    *,
    diagnostic: object | None = None,
) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.INVALID_RESULT,
        f"{component.upper()} returned an invalid result",
        retryable=False,
        diagnostic=diagnostic,
    )


def unsafe_path(
    component: IntegrationComponent,
    *,
    diagnostic: object | None = None,
) -> IntegrationError:
    return IntegrationError(
        component,
        IntegrationErrorCode.UNSAFE_PATH,
        f"{component.upper()} returned an unsafe path",
        retryable=False,
        diagnostic=diagnostic,
    )
