"""
This module contains utility functions to be used in Jupyter notebooks or standard IDEs 
like VSCode, PyCharm, etc. The functions provided here help in interpolating data, 
exporting grids, and calculating sensitivities.

Functions:
-----------

- interpolate(x, y, z, cell_size, method='nearest', smooth_s=0, blank=blank):
    Interpolates scatter data to a regular grid using the specified interpolation method.

- export_grid(grid_in, filename='georaster'):
    Exports the interpolated grid data to a GeoTIFF file.

- lin_sens(geometry, maxdepth=3., n_int=100, sensor_height=0):
    Calculates approximative cumulative and relative sensitivities based on specified geometry.

- update_plot(CLAY, VWC, ECW, BD):
    Updates the plot based on the slider changes.

- update_plot2(CEC, VWC, BD, ECW):
    Updates the plot based on the slider changes.

- waxsmits(vwc, bd, water_ec, CEC, pdn=2.65, m=1.5, n=2, a=0.4):
    Calculates the soil bulk real electrical conductivity using the revised Waxman-Smits model.

- linde(vwc, bd, water_ec, clay, sand, pdn=2.65, m=1.5, n=2):
    Calculates the soil bulk real electrical conductivity using the Linde et al. 2006 model.

- fu(vwc, bd, water_ec, clay, pdn=2.65, solid_ec=(1*(10**-7)), w=2):
    Calculates the soil bulk real electrical conductivity using the Fu et al. 2021 model.

"""
# load the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib
import geopandas as gpd


def interpolate(x, y, z, cell_size, method='nearest', 
                smooth_s=0, blank=None):
    """
    Interpolate scatter data to regular grid through selected interpolation 
    method (with scipy.interpolate for simple interpolation).

    The output of this function is a Numpy array that holds the interpolated 
    data (accessed via `datagrid['grid']` in the cell below), alongside 
    the grid's cell size (`datagrid['cell_size']`) and the grid extent 
    (`datagrid['extent']`). The cell size and extent are needed to allow 
    exporting the interpolated data efficiently to a GeoTIF that can be 
    opened in any GIS software such as QGIS. 

    Parameters
    ----------
    x : np.array
        Cartesian GPS x-coordinates.

    y : np.array
        Cartesian GPS y-coordinates.

    z : np.array
        Data points to interpolate.

    cell_size : float
        Grid cell size (m).

    method : str, optional
        Scipy interpolation method ('nearest', 'linear', 'cubic' or 'IDW')

    smooth_s : float, optional
        Smoothing factor to apply a Gaussian filter on the interpolated grid.
        If 0, no smoothing is performed. (Applying smoothing can result 
        in a loss of detail in the interpolated grid.)

    blank : object, optional
        A blank object to mask (clip) interpolation beyond survey bounds.

    Returns
    -------
    grid : np.array
        Array of interpolated and masked grid containing:
        - the interpolated grid (grid['grid'])
        - the grid cell size (grid['cell_size'])
        - the grid extent (grid['extent'])

    """
    # Define an evenly spaced grid over which the dataset values have to be interpolated
    x_min = x.min()
    x_max = x.max() + cell_size
    y_min = y.min()
    y_max = y.max() + cell_size
    x_vector = np.arange(x_min, x_max, cell_size)
    y_vector = np.arange(y_min, y_max, cell_size)
    extent = (x_vector[0], x_vector[-1], y_vector[0], y_vector[-1])

    xx, yy = np.meshgrid(x_vector, y_vector)
    nx, ny = xx.shape
    coords = np.concatenate((xx.ravel()[np.newaxis].T, 
                        yy.ravel()[np.newaxis].T), 
                        axis=1)
    
    # Create a mask to blank grid outside surveyed area
    boolean = np.zeros_like(xx)
    if blank is not None:
        boundaries = np.vstack(blank.loc[0, 'geometry'].exterior.coords.xy).T
        bound = boundaries.copy()
        boolean += matplotlib.path.Path(
            bound).contains_points(coords).reshape((nx, ny))
    boolean = np.where(boolean >= 1, True, False)
    mask = np.where(boolean == False, np.nan, 1)
    binary = np.where(boolean == False, 0, 1)
    
    # Fast (and sloppy) interpolation (scipy.interpolate)
    if method in ['nearest','cubic', 'linear']:
        # Interpolate 
        data_grid = griddata(
            np.vstack((x, y)).T, z, (xx, yy), method=method
            ) * mask
    else:
        print('define interpolation method')
    
    if smooth_s > 0:
        data_grid = gaussian_filter(data_grid, sigma=smooth_s)

    # Create a structured array with additional fields for coordinates and cell size
    dtype = [
        ('grid', data_grid.dtype, data_grid.shape),
        ('cell_size', float),
        ('extent', [
            ('x_min', float), 
            ('x_max', float), 
            ('y_min', float), 
            ('y_max', float)
            ])
    ]
    grid = np.array((data_grid, cell_size, extent), dtype=dtype)
    
    return grid

# Function to export an interpolated grid as a geotif.
# -------------------------------------------------------

def export_grid(grid_in, filename='georaster', export_path=os.getcwd()): 
    """
    Interpolate scatter data to regular grid through selected interpolation 
    method (with scipy.interpolate for simple interpolation).

    Parameters
    ----------
    grid_in : np.array
        Array of interpolated and masked grid.

    filename : str, optional
        Name of the GeoTIFF (.tif) file (standard = 'gridded').

    export_path : str, optional
        Path to the directory where the GeoTIFF file will be saved (default is current directory).

    Returns
    -------
    data are exported to the specified directory as a GeoTIFF file.
    
    """

    # Get grid properties
    cell_size = grid_in['cell_size']
    extent = grid_in['extent']
    transform = from_origin(extent['x_min'], extent['y_min'], 
                                cell_size, -cell_size)
    
    # Prepare rasterio grid
    grid_exp = grid_in['grid']
    grid_exp[np.isnan(grid_exp)] = -99999
    grid_exp = grid_exp.astype(rasterio.float32)
    nx, ny = grid_exp.shape
    grid_exp = np.flip(grid_exp, axis=0)

    # Create an empty grid with correct name and coordinate system
    output_file = os.path.join(export_path, filename + '.tif')
    with rasterio.open(
        output_file,
        mode='w',
        driver='GTiff',
        height=nx,
        width=ny,
        count=1,
        dtype=str(grid_exp.dtype),
        crs='EPSG:31370', #Lambert 1972 coordinates
        transform=transform,
        nodata=-99999
    ) as dst:
        dst.write(grid_exp, 1)

    # Open the GeoTIFF file in read/write mode to flip the image vertically
    with rasterio.open(output_file, mode='r+') as dst:
        data = dst.read()
        dst.write(data[0, ::-1], 1)

# 0.2b Functions for evaluating LIN depth sensitivity
# ----------------------------------------------------

def lin_sens(geometry, maxdepth=3., n_int=100, sensor_height = 0):
    """
    Calculate approximative cumulative and relative sensitivities based on
    Keller & Frischknecht, 1966 and McNeill, 1980

    Parameters
    ----------
    geometry : str
        coil geometry identifier, combining orientation ('HCP' or 'PRP'),
        and Tx-Rx separation. 'HCP0.5'is thus a HCP orientation with a 0.5 m
        coil separation.

    maxdepth : float (optional, default=3.0)
        Maximum depth to evaluate the sensitivity (m).

    n_int : int (optional, default=100)
        Number of depth intervals to evaluate the sensitivity.

    sensor_height : float (optional, default=0)
        Height of the sensor above the ground (m).

    Returns
    -------
    rsens_QP : np.array
        Array of relative QP sensitivities obtained over the evaluated depths.
    
    csens_QP : np.array, optional
        Array of cumulative QP sensitivities obtained over the evaluated depths.

    rsens_IP : np.array, optional
        Array of relative IP sensitivities obtained over the evaluated depths.
    
    csens_IP : np.array, optional
        Array of cumulative IP sensitivities obtained over the evaluated depths.

    """
    # determine depth extent along which to evaluate sensitivity
    depths = np.linspace(.0, maxdepth, n_int)

    if 'inph' in geometry:
        coil_spacing = float(geometry[3:6])
    else: 
        coil_spacing = float(geometry[-3:])
    
    # create empty arrays
    csens_QP = np.empty_like(depths)
    rsens_QP = np.empty_like(depths)
    csens_IP = np.empty_like(depths)
    rsens_IP = np.empty_like(depths)
    
    depth_ratio = (depths + sensor_height) / coil_spacing
    if 'HCP' in geometry:
        csens_IP = (1 - 8 * depth_ratio ** 2) / ((4 * (depth_ratio ** 2) + 1) ** (5 / 2))
        rsens_IP = 12 * depth_ratio * (3 - 8 * depth_ratio ** 2) / (coil_spacing * ((4 * depth_ratio ** 2) + 1) ** (7 / 2))
        csens_QP = 1 / ((4 * (depth_ratio ** 2) + 1) ** 0.5)
        rsens_QP = 4 * depth_ratio / (coil_spacing * (4 * depth_ratio ** 2 + 1) ** (3 / 2))

    if 'PRP' in geometry:
        csens_IP = (6 * depth_ratio) / ((4 * (depth_ratio ** 2) + 1) ** 2.5)
        rsens_IP = -(96 * (depth_ratio ** 2) - 6) / (coil_spacing * (4 * (depth_ratio ** 2) + 1) ** (7 / 2))
        csens_QP = 1 - ((2 * depth_ratio) / ((4 * depth_ratio ** 2) + 1) ** 0.5)
        rsens_QP = 2 / ((coil_spacing * (4 * depth_ratio ** 2) + 1) ** (3 / 2))

    return rsens_QP, csens_QP, rsens_IP, csens_IP

# Update plot function based on slider changes
def update_plot(CLAY, VWC, ECW, BD, vwc_i, b_dens_i, clay_i):
    # Iterate over bulk density for first plot
    SAND = (100-CLAY)/2
    linde_b_it = [linde(VWC, bd, ECW, CLAY, SAND) for bd in b_dens_i]
    fu_b_it = [fu(VWC, bd, ECW, CLAY) for bd in b_dens_i]
    # Iterate over clay content for second plot
    linde_c_it = [linde(VWC, BD, ECW, clay,((100-clay)/2)) for clay in clay_i]
    fu_c_it = [fu(VWC, BD, ECW, clay) for clay in clay_i]
    # Iterate over VWC for third plot
    linde_v_it = [linde(vwc, BD, ECW, CLAY, SAND) for vwc in vwc_i]
    fu_v_it = [fu(vwc, BD, ECW, CLAY, SAND) for vwc in vwc_i]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(b_dens_i, linde_b_it, linewidth=3)
    axes[0].plot(b_dens_i, fu_b_it, linewidth=3)
    axes[0].set_xlabel("bulk density [g/cm^3]")
    axes[0].set_ylabel("bulk EC [mS/m]")
    axes[0].set_title("change clay and vwc with slider")

    axes[1].plot(clay_i, linde_c_it, linewidth=3)
    axes[1].plot(clay_i, fu_c_it, linewidth=3)
    axes[1].set_xlabel("clay content [%]")
    axes[1].set_ylabel("bulk EC [mS/m]")
    axes[1].set_title("change vwc and bd with slider")

    axes[2].plot(vwc_i, linde_v_it, label='Linde et al., 2006', linewidth=3)
    axes[2].plot(vwc_i, fu_v_it, label='Fu et al., 2021', linewidth=3)
    axes[2].set_xlabel("volumetric water content [%]")
    axes[2].set_ylabel("bulk EC [mS/m]")
    axes[2].set_title("change clay and bd with slider")
    
    axes[2].legend(loc='upper left')
    fig.suptitle(f'Comparison of Linde et al. 2006 and Fu et al. 2021 for EC modelling', fontsize=14)
    plt.show()

# Update plot function
def update_plot2(CEC, VWC, BD,ECW, vwc_i, b_dens_i, CEC_i):
    # Calculate model outputs
    c_v_it = [waxsmits(vwc, BD, ECW, CEC) for vwc in vwc_i]
    c_b_it = [waxsmits(VWC, bd, ECW, CEC) for bd in b_dens_i]
    c_c_it = [waxsmits(VWC, BD, ECW, cec) for cec in CEC_i]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(vwc_i, c_v_it, linewidth=3)
    axes[0].set_xlabel("volumetric water content [%]")
    axes[0].set_ylabel("bulk EC [mS/m]")
    axes[0].set_title("change bd and CEC with slider")

    # CEC and BD plot
    axes[1].plot(b_dens_i, c_b_it, linewidth=3)
    axes[1].set_xlabel("bulk density [g/cm^3]")
    axes[1].set_ylabel("bulk EC [mS/m]")
    axes[1].set_title("change vwc and CEC with slider")


    # CEC and BD plot
    average_ec = np.mean(c_c_it)
    plot_lab = 'mean bulk EC: ' + str(round(average_ec, 2)) + '[mS/m]'
    axes[2].plot(CEC_i, c_c_it, label=plot_lab, linewidth=3)
    
    # set y axis range from 0 to 100 mS/m
    axes[2].set_ylim([0, 100])
    axes[2].set_xlabel("CEC [mmol/100 g]")
    axes[2].set_ylabel("bulk EC [mS/m]")
    axes[2].legend(loc='upper right')
    fig.suptitle('Evaluation of Waxman & Smits (1968) EC model', fontsize=14)
    plt.show()

# 2.0: Pedophysical modelling - Linde et al. 2006
# -----------------------------------------------

def waxsmits(vwc, bd, water_ec, CEC, pdn=2.65, m=1.5, n=2):
    """        
    Pedophysical modelling following the revised Waxman-Smits model

    Parameters
    ----------
    vwc: float
        volumetric water content [%]
    
    bd: float
        bulk density [g/cm3]

    CEC: float
        cation exchange capacity [meq/100g]

    water_ec: float
        Soil water real electrical conductivity [mS/m]

    pdn: float
        particle density [g/cm3]

    m: float
        cementation exponent [-]

    n: float
        saturation exponent [-]

    Returns
    -------
    bulk_ec: float
        Soil bulk real electrical conductivity [mS/m]
    """  
    # porosity and saturation
    por = 1-(bd/pdn)
    sat_w = (vwc/100)/por 

    # Pore water EC // Waxman-Smits notation and units
    Cw_Sm = water_ec/1000 # mS/m to S/m
    Cw_Scm = Cw_Sm / 100.0 # S/m to S/cm (cf. Waxman-Smits, 1968)
    
    # Waxman–Smits Qv: ion exchange capacity per pore volume [meq/cm³ pore volume]
    Q_v = (bd / por) * (CEC / 100.0)

    # formation factor (F* in Revil et al. 2012)
    f_form = por**(-m) 

    # Waxman–Smits equivalent conductance of exchange cations (B [S cm^2/meq]), water conductivity (Cw in S/cm)
    B = 0.046 * (1 - 0.6 * np.exp(-Cw_Scm / 0.013))
    BQv_in_Sm = 100.0 * B * Q_v # S/cm to S/m (cf. Waxman-Smits, 1968)

    bulk_ec_Sm = (sat_w ** n / f_form) * (Cw_Sm + BQv_in_Sm / sat_w)
    return bulk_ec_Sm*1000 # to milliSiemens per meter
    

# def waxsmits_ERROR(vwc, bd, water_ec, CEC, pdn=2.65, m=1.5, n=2, a = 0.4):
#     """        
#     Pedophysical modelling following the revised Waxman-Smits model
    
#     ERROR - identified 2026!!! kept for reference - remove in future versions 
#       confuses Wunderlich & Revil modifications to the original Waxman-Smits model (1968)
#       incorrectly integrates units -- wrong!

#     Parameters
#     ----------
#     vwc: float
#         volumetric water content [%]
    
#     bd: float
#         bulk density [g/cm3]

#     CEC: float
#         cation exchange capacity [meq/100g]

#     water_ec: float
#         Soil water real electrical conductivity [mS/m]

#     pdn: float
#         particle density [g/cm3]

#     m: float
#         cementation exponent [-]

#     n: float
#         saturation exponent [-]

#     a: float
#         fitting parameter [-] (see Wunderlich et al. 2013)

#     Returns
#     -------
#     bulk_ec: float
#         Soil bulk real electrical conductivity [mS/m]
#     """  
#     #Convert particle density to kg/m3
#     pdn_kg = pdn * 1000
#     water_ec = water_ec/1000 # to S/m
#     por = 1-(bd/pdn)
#     sat_w = (vwc/100)/por 
    
#     Q_v = (pdn_kg) * ((1-por) / por) * CEC
#     # formation factor
#     f_form = por**(-m) 
#     # Empirical ion mobility factor (Revil et al. 1998)
#     B = (4.78e-8) * (1 - 0.6 * np.exp(-(water_ec/0.013))) 

#     bulk_ec = (sat_w**n/f_form) / a * (water_ec + ((B * Q_v)/sat_w))
#     return bulk_ec*1000 # to milliSiemens per meter
    

def linde(vwc, bd, water_ec, clay, sand, pdn=2.65, m=1.5, n=2):
    """        
    Pedophysical modelling following the Linde et al. 2006 model.

    Parameters
    ----------
    vwc: float
        volumetric water content [%]
    
    bd: float
        bulk density [g/cm3]

    clay: float
        Soil volumetric clay content [%]

    water_ec: float
        Soil water real electrical conductivity [mS/m]

    pdn: float
        particle density [g/cm3]

    m: float
        cementation exponent [-]

    n: float
        saturation exponent [-]

    Returns
    -------
    bulk_ec: float
        Soil bulk real electrical conductivity [mS/m]
    """  

    por = 1-(bd/pdn) # porosity
    sat_w = (vwc/100)/por # water saturation
    f_form = por**(-m) # formation factor
    
    water_ec = water_ec/1000

    silt = 100 - clay - sand

    radius_clay = 0.002/2000
    radius_silt = 0.025/2000
    radius_sand = 0.75/2000

    solid_ec = 1*(10**-7) # Solid electrical conductivity
    clay_ec= 3*(solid_ec/radius_clay)  # clay electrical conductivity
    silt_ec = 3*(solid_ec/radius_silt) # Silt electrical conductivity
    sand_ec = 3*(solid_ec/radius_sand) # Sand electrical conductivity

    surf_ec = np.average([clay_ec*(clay/100), 
                          sand_ec*(sand/100), 
                          silt_ec*(silt/100)])
    bulk_ec = (((sat_w**n)*water_ec) 
               + (f_form - 1)*(surf_ec))/f_form 
    
    return bulk_ec*1000

def fu(vwc, bd, water_ec, clay, pdn=2.65, solid_ec = (1*(10**-7)), w=2):
    """
    Pedophysical modelling following Fu et al. 2021

    Parameters
    ----------
    vwc : float
        Soil volumetric water content [%].

    bd : float
        Soil bulk density [g/cm3].

    water_ec : float
        Soil water real electrical conductivity [mS/m].

    clay : float
        Soil clay content [%].

    pdn : float, optional
        Particle density [g/cm3], default is 2.65.

    solid_ec : float, optional
        Soil solid real electrical conductivity [mS/m], default is 1e-7.
        
    w : float, optional
        Phase exponent of the water, default is 2.
    
    Returns
    -------
    bulk_ec: float
        Soil bulk real electrical conductivity [mS/m]

    """
    d = 0.6539
    e = 0.0183
    vwc = vwc/100 # to m3/m3
    water_ec  = water_ec/1000 # S/m
    surf_ec = (d*clay/(100-clay))+e # S/m
    por = 1-(bd/pdn) # porosity []

    bulk_ec = water_ec * vwc**w + vwc * por * surf_ec + (1-por) * solid_ec
    
    return bulk_ec*1000
