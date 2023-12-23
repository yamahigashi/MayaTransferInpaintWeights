# # -*- coding: utf-8 -*-
import sys
import textwrap

from maya import (
    cmds,
    mel,
)


if sys.version_info > (3, 0):
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import (
            Optional,  # noqa: F401
            Dict,  # noqa: F401
            List,  # noqa: F401
            Tuple,  # noqa: F401
            Pattern,  # noqa: F401
            Callable,  # noqa: F401
            Any,  # noqa: F401
            Text,  # noqa: F401
            Generator,  # noqa: F401
            Union,  # noqa: F401
            Iterable # noqa: F401
        )


##############################################################################
def __register_menu():
    """Setup menu"""

    from textwrap import dedent
    cmds.evalDeferred(dedent(
        """
        try:
            import robust_skin_weights_transfer_inpaint.setup as setup
            setup.register_menu()
        except:
            import traceback
            traceback.print_exc()
        """
    ))


if __name__ == '__main__':
    try:
        __register_menu()

    except Exception:
        import traceback
        traceback.print_exc()
