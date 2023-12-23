# # -*- coding: utf-8 -*-
import os
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
def register_menu():
    # type: () -> None
    """Setup menu"""

    if cmds.about(batch=True):
        return

    if not is_default_window_menu_registered():
        register_default_window_menu()

    cmds.setParent("MayaWindow|mainWindowMenu", menu=True)

    cmds.menuItem(divider=True)
    item = cmds.menuItem(
        "open_skin_weight_transfer_inpaint_window",
        label="Skin Weight Transfer Inpaint",
        annotation="Show Skin Weight Transfer Inpaint Window",
        echoCommand=True,
        command=textwrap.dedent(
            """
                import robust_skin_weights_transfer_inpaint.setup as setup
                if setup.check_dependencies():
                    import robust_skin_weights_transfer_inpaint.ui as ui
                ui.show()
            """)
    )
    print("TransferInpaintWeights: register menu item as {}".format(item))


def is_default_window_menu_registered():
    # type: () -> bool
    """Check if default Window menu is registered"""
    if not cmds.menu("MayaWindow|mainWindowMenu", exists=True):
        return False

    kids = cmds.menu("MayaWindow|mainWindowMenu", query=True, itemArray=True)
    if not kids:
        return False

    if len(kids) == 0:
        return False

    return True


def register_default_window_menu():
    cmd = '''
    buildViewMenu MayaWindow|mainWindowMenu;
    setParent -menu "MayaWindow|mainWindowMenu";
    '''

    mel.eval(cmd)


def deregister_menu():
    # type: () -> None
    """Remove menu"""

    if cmds.about(batch=True):
        return

    try:
        path = "MayaWindow|mainWindowMenu|open_skin_weight_transfer_inpaint_window"
        cmds.deleteUI(path, menuItem=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)


def check_python_version():
    if sys.version_info < (3, 6):
        print("Python 3.6 or higher is required.")
        sys.exit(1)


def check_dependencies():
    executable = os.path.join(os.path.dirname(sys.executable), "mayapy")
    is_valid = True

    try:
        import numpy

    except ImportError:
        is_valid = False
        title = "Missing dependency Please install numpy first"
        command = '"{}" -m pip install numpy'.format(executable)
        message = "Please install numpy first" \
                " by running the following command:\n\n"

        show_error(title, message, command)

    try:
        import scipy
    except ImportError:
        is_valid = False
        title = "Missing dependency Please install scipy first"
        command = '"{}" -m pip install scipy'.format(executable)
        message = "Please install scipy first" \
                " by running the following command in a terminal:\n\n"

        show_error(title, message, command)

    return is_valid


def show_error(title, message, command):
    import maya.cmds as cmds
    cmds.promptDialog(
            title=title,
            message=message,
            text=command,
            button="OK",
            defaultButton="OK")
