!pip install "tidy3d[trimesh]"
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web
import trimesh
web.configure("") 


wl_c = 1.55
wl_bw = 0.100
wl_n = 101

mat_psio2 = td.Medium(permittivity=1.15**2)
mat_air = td.Medium(permittivity=1.00)
mat_ipdip = td.Medium(permittivity=1.57**2)
mat_fiber = td.Medium(permittivity=1.46**2)

lens_d = 15
#lens_f = 10.27
lens_f = 10.5
cone_h = 5
cone_z = -lens_f-cone_h
cone_n_sections = 100
wg_d = 1.2
wg_z = cone_z-3
fiber_z = 10.27
fiber_d = 8

pad_x = 1 * wl_c
pad_y = 1.5 * wl_c
wl_range = np.linspace(wl_c - wl_bw / 2, wl_c + wl_bw / 2, wl_n)
freq_c = td.C_0 / wl_c
freq_range = td.C_0 / wl_range
freq_width = 0.5 * (np.max(freq_range) - np.min(freq_range))
run_time = 30 / freq_width
_inf = 1e3

def get_simulations():

  modesource_0 = td.ModeSource(

      name = 'modesource_0',
      center = [0, 0, fiber_z+3],
      size = [fiber_d+2, fiber_d+2, 0],
      source_time = td.GaussianPulse(freq0 = freq_c, fwidth =freq_width ),
      direction = '-',
      num_freqs=7,
  )

  permittivitymonitor_0 = td.PermittivityMonitor(
      name = 'permittivitymonitor_0',
      size = [0, 16, 50],
      freqs = [freq_c],
  )

  field_yz = td.FieldMonitor(
      name = 'field_yz',
      size = [0, 16, 50],
      freqs = [freq_c],
  )

  field_input = td.FieldMonitor(
      name = 'field_input',
      center = [0, 0, fiber_z+3],
      size = [fiber_d+2, fiber_d+2, 0],
      freqs = [freq_c],
  )

  field_output = td.FieldMonitor(
      name = 'field_output',
      center = [0, 0, wg_z],
      size = [wg_d+1, wg_d+1, 0],
      freqs = [freq_c],
  )

  mode_spec = td.ModeSpec(num_modes=1)
  mode_monitor = td.ModeMonitor(
      center = (0,0,wg_z),
      size = (wg_d+1,wg_d+1,0),
      freqs = freq_range,
      mode_spec=mode_spec,
      name="mode_0",
  )

# fluxmonitor_2 = td.FluxMonitor(
#     name = 'fluxmonitor_2',
#     center = [0, 0, -20],
#     size = [20, 20, 0],
#     freqs = [199861643525342.66, 196585223139681.3, 193414493734202.6, 190344422405088.28, 187370290805008.75],
# )

  psio2_medium = td.Structure(
      geometry = td.Box.from_bounds(
          rmin=(-20,-20,-30),rmax = (20,20,8)
      ),
      medium=mat_psio2,
  )


  ball_lens = td.Structure(
      geometry = td.Sphere(radius = lens_d/2, ),
      name = 'sphere_0',
      medium = mat_ipdip,
  )

  fiber = td.Structure(
      geometry = td.Cylinder(radius = fiber_d/2, center = [0, 0, fiber_z+5], length = 10),
      name = 'fiber',
      medium = mat_fiber,
  )

  waveguide = td.Structure(
      geometry = td.Cylinder(radius = wg_d/2, center = [0, 0, wg_z], length = 6),
      name = 'waveguide',
      medium = mat_ipdip
  )
  #just taper
  # Create a cone mesh.
  cone_mesh = trimesh.creation.cone(radius = wg_d/2, height = cone_h, sections = cone_n_sections)
  cone_mesh.apply_translation([0,0,cone_z])
  cone_geo = td.TriangleMesh.from_trimesh(cone_mesh)
  cone = td.Structure(geometry=cone_geo, medium=mat_ipdip)
  sim_array = []
  sim_array.append(
      td.Simulation(
          size = [20, 20, fiber_z-wg_z+2],
          center = [0,0,(fiber_z+wg_z)/2],
    # grid_spec = td.GridSpec(wavelength = 1.55, ),
          version = '2.6.4',
          run_time = run_time,
          medium = mat_air,
          sources = [modesource_0],
          grid_spec=td.GridSpec.auto(min_steps_per_wvl=15, wavelength=wl_c),
          monitors = [permittivitymonitor_0,field_yz,field_input,field_output,mode_monitor],
          structures = [psio2_medium,ball_lens,fiber,waveguide,cone],
          boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML(num_layers=20)),
    symmetry=(1, -1, 0),
      )

  )
  sims = {sim_name:sim for sim_name, sim in zip(['normal_simulation'],sim_array)}

sim_array = get_simulations()
batch = web.Batch(simulations=sim_array, verbose=True)
tot_cost = 0
for bat in batch.get_info().values():
    sim_name = bat.taskName
    cost = web.estimate_cost(bat.taskId, verbose=False)
    tot_cost += cost
    print(f"Maximum FlexCredit cost for {sim_name} = {cost:.2f}")
print(f"Maximum FlexCredit cost for batch = {tot_cost:.2f}")
batch_array = batch.run(path_dir="data/data_array")
sim_array_results = {tl: sn for tl, sn in batch_array.items()}



power_array_25 = []
for sim_data in sim_array_results.values():
    mode_amps = sim_data["mode_0"]
    coeffs_f = mode_amps.amps.sel(direction="-")
    power_0 = np.abs(coeffs_f.sel(mode_index=0)) ** 2
    power_0_db = 10 * np.log10(power_0)
    power_array_25.append(power_0_db)
    # mode_amps1 = sim_data["field_output"]

plot_colors = ("black", "red", "blue")
fig, ax1 = plt.subplots(1, figsize=(8, 4))
for data, color, label in zip(power_array_25, plot_colors, sim_array_results.keys()):
    ax1.plot(
        wl_range,
        data,
        color=color,
        linestyle="solid",
        linewidth=1.0,
        label=label,
    )
ax1.set_xlim([wl_range[0], wl_range[-1]])
ax1.set_xlabel("Wavelength (um)")
ax1.set_ylabel("Power (dB)")
ax1.set_title("Coupling Efficiency")
plt.legend()
plt.show()
  return sims
