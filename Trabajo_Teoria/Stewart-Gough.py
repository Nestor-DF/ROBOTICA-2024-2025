import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def plot_stewart_gough(base_points, platform_points, ax):
    ax.clear()

    # Dibujar la base
    base_x, base_y, base_z = zip(*base_points)
    ax.plot(
        base_x + (base_x[0],),
        base_y + (base_y[0],),
        base_z + (base_z[0],),
        "b-o",
        label="Base",
    )

    # Dibujar la plataforma
    platform_x, platform_y, platform_z = zip(*platform_points)
    ax.plot(
        platform_x + (platform_x[0],),
        platform_y + (platform_y[0],),
        platform_z + (platform_z[0],),
        "r-o",
        label="Plataforma",
    )

    # Conectar la base con la plataforma
    for bp, pp in zip(base_points, platform_points):
        ax.plot([bp[0], pp[0]], [bp[1], pp[1]], [bp[2], pp[2]], "g--")

    # Configuración del gráfico
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Plataforma Stewart-Gough")
    ax.legend()

    # Fijar límites de los ejes
    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-6, 6])


# Generar puntos de base y plataforma
def generate_points(radius, height, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [(radius * np.cos(a), radius * np.sin(a), height) for a in angles]
    return points


# Aplicar transformación a los puntos de la plataforma
def transform_platform_points(platform_points, translation, rotation):
    rotation_matrix = np.array(
        [
            [
                np.cos(rotation[2]) * np.cos(rotation[1]),
                np.cos(rotation[2]) * np.sin(rotation[1]) * np.sin(rotation[0])
                - np.sin(rotation[2]) * np.cos(rotation[0]),
                np.cos(rotation[2]) * np.sin(rotation[1]) * np.cos(rotation[0])
                + np.sin(rotation[2]) * np.sin(rotation[0]),
            ],
            [
                np.sin(rotation[2]) * np.cos(rotation[1]),
                np.sin(rotation[2]) * np.sin(rotation[1]) * np.sin(rotation[0])
                + np.cos(rotation[2]) * np.cos(rotation[0]),
                np.sin(rotation[2]) * np.sin(rotation[1]) * np.cos(rotation[0])
                - np.cos(rotation[2]) * np.sin(rotation[0]),
            ],
            [
                -np.sin(rotation[1]),
                np.cos(rotation[1]) * np.sin(rotation[0]),
                np.cos(rotation[1]) * np.cos(rotation[0]),
            ],
        ]
    )

    transformed_points = []
    for point in platform_points:
        rotated_point = np.dot(rotation_matrix, point)
        translated_point = rotated_point + translation
        transformed_points.append(translated_point)

    return transformed_points


# Parámetros
num_legs = 6
base_radius = 5
platform_radius = 3
base_height = 0
platform_height = 3

# Generar puntos de la base y de la plataforma
base_points = generate_points(base_radius, base_height, num_legs)
platform_points = generate_points(platform_radius, platform_height, num_legs)


# Control manual de los 6 grados de libertad
class ControlPanel:
    def __init__(self, root, update_callback):
        self.translation_x = tk.DoubleVar(value=0.0)
        self.translation_y = tk.DoubleVar(value=0.0)
        self.translation_z = tk.DoubleVar(value=0.0)
        self.rotation_roll = tk.DoubleVar(value=0.0)
        self.rotation_pitch = tk.DoubleVar(value=0.0)
        self.rotation_yaw = tk.DoubleVar(value=0.0)
        self.update_callback = update_callback

        self.create_controls(root)

    def create_controls(self, root):
        frame = ttk.Frame(root)
        frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(frame, text="Translation X").pack()
        ttk.Scale(
            frame,
            variable=self.translation_x,
            from_=-5,
            to=5,
            orient="horizontal",
            length=300,
            command=self.update_callback,
        ).pack()

        ttk.Label(frame, text="Translation Y").pack()
        ttk.Scale(
            frame,
            variable=self.translation_y,
            from_=-5,
            to=5,
            orient="horizontal",
            length=300,
            command=self.update_callback,
        ).pack()

        ttk.Label(frame, text="Translation Z").pack()
        ttk.Scale(
            frame,
            variable=self.translation_z,
            from_=-5,
            to=5,
            orient="horizontal",
            length=300,
            command=self.update_callback,
        ).pack()

        ttk.Label(frame, text="Rotation Roll").pack()
        ttk.Scale(
            frame,
            variable=self.rotation_roll,
            from_=-180,
            to=180,
            orient="horizontal",
            length=300,
            command=self.update_callback,
        ).pack()

        ttk.Label(frame, text="Rotation Pitch").pack()
        ttk.Scale(
            frame,
            variable=self.rotation_pitch,
            from_=-180,
            to=180,
            orient="horizontal",
            length=300,
            command=self.update_callback,
        ).pack()

        ttk.Label(frame, text="Rotation Yaw").pack()
        ttk.Scale(
            frame,
            variable=self.rotation_yaw,
            from_=-180,
            to=180,
            orient="horizontal",
            length=300,
            command=self.update_callback,
        ).pack()

        ttk.Button(frame, text="Reset", command=self.reset_controls).pack(pady=10)

    def reset_controls(self):
        self.translation_x.set(0.0)
        self.translation_y.set(0.0)
        self.translation_z.set(0.0)
        self.rotation_roll.set(0.0)
        self.rotation_pitch.set(0.0)
        self.rotation_yaw.set(0.0)
        self.update_callback()


# Crear ventana principal
root = tk.Tk()
root.title("Control de Plataforma Stewart-Gough")

# Crear figura de Matplotlib embebida en Tkinter
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Función de actualización
platform_points_transformed = platform_points


def update_simulation(event=None):
    global platform_points_transformed
    translation = np.array(
        [
            control_panel.translation_x.get(),
            control_panel.translation_y.get(),
            control_panel.translation_z.get(),
        ]
    )
    rotation = np.radians(
        [
            control_panel.rotation_roll.get(),
            control_panel.rotation_pitch.get(),
            control_panel.rotation_yaw.get(),
        ]
    )

    platform_points_transformed = transform_platform_points(
        np.array(platform_points), translation, rotation
    )
    plot_stewart_gough(base_points, platform_points_transformed, ax)
    canvas.draw()


# Crear panel de control
control_panel = ControlPanel(root, update_simulation)
update_simulation()

root.mainloop()
