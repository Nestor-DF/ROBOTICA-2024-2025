import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Entry, Button, Frame

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

# Calcular longitudes de los actuadores (cinemática inversa)
def calculate_leg_lengths(base_points, platform_points):
    lengths = []
    for bp, pp in zip(base_points, platform_points):
        length = np.sqrt((pp[0] - bp[0])**2 + (pp[1] - bp[1])**2 + (pp[2] - bp[2])**2)
        lengths.append(length)
    return lengths

# Visualización de la plataforma
def plot_stewart_gough(ax, base_points, platform_points, lengths):
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
    for bp, pp, length in zip(base_points, platform_points, lengths):
        ax.plot([bp[0], pp[0]], [bp[1], pp[1]], [bp[2], pp[2]], "g--")
        ax.text((bp[0]+pp[0])/2, (bp[1]+pp[1])/2, (bp[2]+pp[2])/2, f"{length:.2f}")

    # Configuración del gráfico
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Plataforma Stewart-Gough")
    ax.legend()

# Parámetros
num_legs = 6
base_radius = 5
platform_radius = 3
base_height = 0
platform_height = 3

# Generar puntos de la base y de la plataforma
base_points = generate_points(base_radius, base_height, num_legs)
platform_points = generate_points(platform_radius, platform_height, num_legs)

# Crear ventana principal
root = Tk()
root.title("Cinemática Inversa de la Plataforma Stewart-Gough")

# Crear marco para controles
control_frame = Frame(root)
control_frame.pack(side="left", padx=10, pady=10)

# Crear etiquetas y entradas para la posición
Label(control_frame, text="Posición X:").grid(row=0, column=0)
entry_x = Entry(control_frame)
entry_x.grid(row=0, column=1)

Label(control_frame, text="Posición Y:").grid(row=1, column=0)
entry_y = Entry(control_frame)
entry_y.grid(row=1, column=1)

Label(control_frame, text="Posición Z:").grid(row=2, column=0)
entry_z = Entry(control_frame)
entry_z.grid(row=2, column=1)

# Crear etiquetas y entradas para la orientación
Label(control_frame, text="Rotación Roll (°):").grid(row=3, column=0)
entry_roll = Entry(control_frame)
entry_roll.grid(row=3, column=1)

Label(control_frame, text="Rotación Pitch (°):").grid(row=4, column=0)
entry_pitch = Entry(control_frame)
entry_pitch.grid(row=4, column=1)

Label(control_frame, text="Rotación Yaw (°):").grid(row=5, column=0)
entry_yaw = Entry(control_frame)
entry_yaw.grid(row=5, column=1)

# Crear botón para calcular y actualizar
canvas = None
fig = None
ax = None

def calculate_and_plot():
    global ax
    # Obtener valores de entrada
    x = float(entry_x.get())
    y = float(entry_y.get())
    z = float(entry_z.get())
    roll = np.radians(float(entry_roll.get()))
    pitch = np.radians(float(entry_pitch.get()))
    yaw = np.radians(float(entry_yaw.get()))

    # Transformar puntos de la plataforma
    translation = np.array([x, y, z])
    rotation = np.array([roll, pitch, yaw])
    transformed_platform_points = transform_platform_points(
        platform_points, translation, rotation
    )

    # Calcular longitudes de los actuadores
    lengths = calculate_leg_lengths(base_points, transformed_platform_points)

    # Mostrar resultados
    plot_stewart_gough(ax, base_points, transformed_platform_points, lengths)
    canvas.draw()

def reset_values():
    entry_x.delete(0, 'end')
    entry_y.delete(0, 'end')
    entry_z.delete(0, 'end')
    entry_roll.delete(0, 'end')
    entry_pitch.delete(0, 'end')
    entry_yaw.delete(0, 'end')
    entry_x.insert(0, "0")
    entry_y.insert(0, "0")
    entry_z.insert(0, "0")
    entry_roll.insert(0, "0")
    entry_pitch.insert(0, "0")
    entry_yaw.insert(0, "0")
    calculate_and_plot()

Button(control_frame, text="Calcular y Mostrar", command=calculate_and_plot).grid(row=6, columnspan=2)
Button(control_frame, text="Reset", command=reset_values).grid(row=7, columnspan=2)

# Crear figura para el gráfico
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side="right", fill="both", expand=True)

# Iniciar bucle de la interfaz
def initialize():
    entry_x.insert(0, "0")
    entry_y.insert(0, "0")
    entry_z.insert(0, "0")
    entry_roll.insert(0, "0")
    entry_pitch.insert(0, "0")
    entry_yaw.insert(0, "0")
    calculate_and_plot()

initialize()
root.mainloop()
