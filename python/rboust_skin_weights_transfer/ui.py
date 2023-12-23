"""This module contains the UI for the application."""
import sys

from maya import cmds
from maya.app.general.mayaMixin import MayaQWidgetBaseMixin

from Qt.QtWidgets import (  # type: ignore
    QApplication,
    QWidget,
    QLineEdit,
    QPushButton,
    QLabel,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QRadioButton,
    QGroupBox,
    QSpacerItem,
    QSizePolicy,
)

from Qt.QtCore import (
    Qt,
    QSize,
)

from rboust_skin_weights_transfer import logic


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
            Union  # noqa: F401
        )


########################################################################################################################


class FloatSlider(QWidget):
    def __init__(self, label, minimum=0.0, maximum=1.0, interval=0.05, step=0.001, initial_value=0.05):
        # type: (str, float, float, float, float, float) -> None

        super(FloatSlider, self).__init__()

        self.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.sizePolicy().setVerticalPolicy(QSizePolicy.MinimumExpanding)
        self.sizeHint()

        self.minimum = minimum
        self.maximum = maximum
        self.interval = interval
        self.step = step
        self.value_multiplier = 1.0 / step

        self.initUI(label, initial_value)

    def initUI(self, label, initial_value):
        # Create the slider and the label
        self.label = QLabel(label, self)
        self.label.setFixedWidth(50)
        self.label.setAlignment(Qt.AlignRight)
        self.value_display = QLabel("{:.2f}".format(initial_value), self)
        self.slider = QSlider(Qt.Horizontal, self)
        
        # Set the range and step of the slider
        self.slider.setMinimum(self.minimum * self.value_multiplier)
        self.slider.setMaximum(self.maximum * self.value_multiplier)
        self.slider.setTickInterval(self.interval * self.value_multiplier)
        self.slider.setSingleStep(self.step * self.value_multiplier)
        self.slider.setValue(initial_value * self.value_multiplier)
        
        # Connect the valueChanged signal to the slot
        self.slider.valueChanged.connect(self.updateValueDisplay)

        # Create the layout and add the widgets
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_display)
        self.setLayout(layout)

    def sizeHint(self):
        # type: () -> QSize
        """Return the size hint of the widget."""
        return QSize(300, 30)

    def updateValueDisplay(self, value):
        # Calculate the float value based on the slider's integer value
        float_value = value / self.value_multiplier
        self.value_display.setText("{:.2f}".format(float_value))

    def value(self):
        # Get the current float value of the slider
        return self.slider.value() / self.value_multiplier

    def setValue(self, float_value):
        # Set the value of the slider using a float
        self.slider.setValue(int(float_value * self.value_multiplier))


class WeightTransferInpaintMainWidget(MayaQWidgetBaseMixin, QWidget):

    def __init__(self, parent=None):
        # type: (QWidget|None) -> None
        super(WeightTransferInpaintMainWidget, self).__init__(parent)

        self.unconfident_vertices = []  # type: List[int]
        self.initUI()

    def initUI(self):
        # type: () -> None

        # grid = QGridLayout()

        # -----------------------------------------------
        # source and destination meshes
        self.meshes_group_box = QGroupBox("Meshes", self)
        self.src_label = QLabel("src:", self)
        self.src_label.setAlignment(Qt.AlignRight)
        self.src_label.setFixedWidth(60)
        self.src_line_edit = QLineEdit(self)
        self.src_line_edit.setReadOnly(True)
        self.src_line_edit.setFocusPolicy(Qt.NoFocus)
        self.src_button = QPushButton("set", self)
        self.src_button.clicked.connect(self.insertText)
        
        self.dst_label = QLabel("dst:", self)
        self.dst_label.setAlignment(Qt.AlignRight)
        self.dst_label.setFixedWidth(60)
        self.dst_line_edit = QLineEdit(self)
        self.dst_line_edit.setReadOnly(True)
        self.dst_line_edit.setFocusPolicy(Qt.NoFocus)
        self.dst_button = QPushButton("set", self)
        self.dst_button.clicked.connect(self.insertText)

        # -----------------------------------------------
        # search settings
        self.search_group_box = QGroupBox("Search Settings", self)
        self.mode_label = QLabel("mode:", self)
        self.mode_label.setAlignment(Qt.AlignRight)
        self.mode_label.setFixedWidth(60)
        self.dist_slider = FloatSlider("distance:", minimum=0.000001, maximum=1.0, interval=0.1, step=0.005, initial_value=0.05)
        self.angle_slider = FloatSlider("angle:", minimum=0.0, maximum=180.0, interval=1.0, step=0.5, initial_value=25.0)

        self.radio_accurate = QRadioButton("Accurate (Slow)", self)
        self.radio_fast = QRadioButton("Inaccurate (Fast)", self)
        self.radio_fast.setChecked(True)
        self.radio_accurate.toggled.connect(self.radioButtonToggled)
        self.radio_fast.toggled.connect(self.radioButtonToggled)

        # -----------------------------------------------
        self.vertex_count_label = QLabel("vertex count:", self)
        self.vertex_count_label.setAlignment(Qt.AlignRight)
        self.vertex_count_label.setFixedWidth(70)
        self.confident_count_display = QLabel("0", self)
        self.unconfident_count_display = QLabel("0", self)

        # -----------------------------------------------
        self.search_button1 = QPushButton("search", self)
        self.search_button1.clicked.connect(self.searchButtonClicked)

        self.select_button1 = QPushButton("select", self)
        self.select_button1.clicked.connect(self.selectButtonClicked)
        
        self.transfer_button1 = QPushButton("transfer", self)
        self.transfer_button1.clicked.connect(self.transferButtonClicked)

        self.inpaint_button1 = QPushButton("inpaint", self)
        self.inpaint_button1.clicked.connect(self.inpaintButtonClicked)
        
        # -----------------------------------------------
        # Create layouts
        src_layout = QHBoxLayout()
        src_layout.addWidget(self.src_label)
        src_layout.addWidget(self.src_line_edit)
        src_layout.addWidget(self.src_button)

        dst_layout = QHBoxLayout()
        dst_layout.addWidget(self.dst_label)
        dst_layout.addWidget(self.dst_line_edit)
        dst_layout.addWidget(self.dst_button)

        meshes_group_box_layout = QVBoxLayout()
        meshes_group_box_layout.addLayout(src_layout)
        meshes_group_box_layout.addLayout(dst_layout)
        self.meshes_group_box.setLayout(meshes_group_box_layout)

        dist_layout = QHBoxLayout()
        dist_layout.addWidget(self.dist_slider)

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(self.angle_slider)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.mode_label)
        mode_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        mode_layout.addWidget(self.radio_accurate)
        mode_layout.addWidget(self.radio_fast)
        mode_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        search_group_box_layout = QVBoxLayout()
        search_group_box_layout.addLayout(dist_layout)
        search_group_box_layout.addLayout(angle_layout)
        search_group_box_layout.addLayout(mode_layout)
        self.search_group_box.setLayout(search_group_box_layout)

        count_layout = QHBoxLayout()
        count_layout.addWidget(self.vertex_count_label)
        count_layout.addWidget(QLabel("confident:"))
        count_layout.addWidget(self.confident_count_display)
        count_layout.addWidget(QLabel("unconfident:"))
        count_layout.addWidget(self.unconfident_count_display)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.search_button1)
        buttons_layout.addWidget(self.select_button1)
        buttons_layout.addWidget(self.transfer_button1)
        buttons_layout.addWidget(self.inpaint_button1)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.meshes_group_box)
        main_layout.addWidget(self.search_group_box)
        main_layout.addLayout(count_layout)
        spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        # Set window properties
        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle("Transfer and Inpainting Skinning Weights")
        self.show()

    def insertText(self):
        # type: () -> None
        """Insert text into the line edit."""

        selection = cmds.ls(sl=True, objectsOnly=True)
        if not selection:
            cmds.warning("Nothing is selected")
            return

        sel = selection[0]

        sender = self.sender()
        if sender == self.src_button:

            if self.src_line_edit.text() != sel:
                self.clear()

            self.src_line_edit.setText(sel)

        elif sender == self.dst_button:

            if self.dst_line_edit.text() != sel:
                self.clear()

            self.dst_line_edit.setText(sel)

    def clear(self):
        # type: () -> None
        self.unconfident_vertices = []
        self.confident_count_display.setText("0")
        self.unconfident_count_display.setText("0")

    def updateValueDisplay(self):
        # type: () -> None
        """Update the value display label."""

        sender = self.sender()
        if sender == self.dist_slider:
            self.dist_value_display.setText(str(self.dist_slider.value()))

        elif sender == self.angle_slider:
            self.angle_value_display.setText(str(self.angle_slider.value()))

    def radioButtonToggled(self):
        if self.radio_accurate.isChecked():
            pass
        elif self.radio_fast.isChecked():
            pass

    def getSearchMode(self):
        # type: () -> str
        """Return the search mode."""

        if self.radio_accurate.isChecked():
            return "accurate"
        elif self.radio_fast.isChecked():
            return "fast"
        else:
            raise RuntimeError("No search mode selected")


    def searchButtonClicked(self):
        # type: () -> None
        """Search for vertices to transfer weights from."""

        src = self.src_line_edit.text()
        dst = self.dst_line_edit.text()
        dist = self.dist_slider.value()
        angle = self.angle_slider.value()
        mode = self.getSearchMode()

        use_fast_search = True if mode == "fast" else False

        if not src or not dst:
            cmds.warning("Source and destination meshes must be set")
            return

        if src == dst:
            cmds.warning("Source and destination meshes must be different")
            return

        if not cmds.objExists(src) or not cmds.objExists(dst):
            cmds.warning("Source and destination meshes must exist")
            return

        # Get the vertices to transfer weights from
        src_mesh, dst_mesh = logic.load_meshes(src, dst)
        tmp = logic.segregate_vertices_by_confidence(src_mesh, dst_mesh, dist, angle, use_fast_search)
        confident, self.unconfident_vertices = tmp

        # Update the display
        self.confident_count_display.setText(str(len(confident)))
        self.unconfident_count_display.setText(str(len(self.unconfident_vertices)))

    def selectButtonClicked(self):
        # type: () -> None
        """Select the vertices to transfer weights from."""

        if not self.unconfident_vertices:
            cmds.warning("No vertices to select")
            return

        dst = self.dst_line_edit.text()
        cmds.select(clear=True)
        cmds.select(["{}.vtx[{}]".format(dst, v) for v in self.unconfident_vertices], add=True)

    def transferButtonClicked(self):
        # type: () -> None
        """Transfer weights from the vertices to transfer weights from."""

        src = self.src_line_edit.text()
        dst = self.dst_line_edit.text()

        if not src or not dst:
            cmds.warning("Source and destination meshes must be set")
            return

        if src == dst:
            cmds.warning("Source and destination meshes must be different")
            return

        if not cmds.objExists(src) or not cmds.objExists(dst):
            cmds.warning("Source and destination meshes must exist")
            return

        # Transfer the weights
        logic.transfer_weights(src, dst, self.unconfident_vertices)

    def inpaintButtonClicked(self):
        # type: () -> None
        """Inpaint the vertices to transfer weights from."""

        if not self.unconfident_vertices:
            cmds.warning("No vertices to inpaint")
            return

        src = self.src_line_edit.text()
        dst = self.dst_line_edit.text()

        if not src or not dst:
            cmds.warning("Source and destination meshes must be set")
            return

        if src == dst:
            cmds.warning("Source and destination meshes must be different")
            return

        if not cmds.objExists(src) or not cmds.objExists(dst):
            cmds.warning("Source and destination meshes must exist")
            return

        try:
            logic.get_skincluster(dst)
        except RuntimeError:
            logic.transfer_weights(src, dst)

        # Inpaint the weights
        cmds.progressBar(progress=0, edit=True, beginProgress=True, isInterruptable=True)
        logic.inpaint_weights(dst, self.unconfident_vertices)
        cmds.progressBar(progress=100, edit=True, endProgress=True)
        cmds.inViewMessage(amg="Inpainting complete", pos="topCenter", fade=True, alpha=0.9)


def show():
    # type: () -> None
    """Show the UI."""
    global MAIN_WIDGET

    # close all previous windows
    all_widgets = {w.objectName(): w for w in QApplication.allWidgets()}
    for k, v in all_widgets.items():
        if v.__class__.__name__ == "WeightTransferInpaintMainWidget":
            v.close()

    main_widget = WeightTransferInpaintMainWidget()
    main_widget.show()

    MAIN_WIDGET = main_widget


if __name__ == "__main__":
    show()
