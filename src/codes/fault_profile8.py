import sys
import os
import numpy as np
import time
from scipy.ndimage import map_coordinates
import shapefile
from shapely.geometry import LineString, Polygon, MultiLineString
import rasterio
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# this script creates a single fault-perpendicular stacked profile across a displacement map (with EW and NS displacement components, e.g. from COSI-Corr or MiMac). # Make sure the sign convention is correct for MM NS-component.
# inputs are a shapefile for the fault rupture (must be a single continuous fault trace)
# this script will interpolate the rupture location where it crosses every pixel in the displacement map
# there's some optical smoothing of the rupture, to simplify local roughness
# I removed the options for comparing displacement profiles from different datasets, bu that can be added back in (it's why we loop over the single number 5!)
# Important things which need adding:
# 1. we need to extend the fault beyond the end points, so we can compute the displacement right up to the end of the ruptures (maybe this isn't too important if we're using a small swatch width)
# 2. we need to update so we create many swatch profiles along the entire fault strike
# 3. currently there are no options for automatically estimating the displacement (e.g. linear regression from each side of the fualt, and then extrpolating these fits onto the fault location, and calculating the offset on the fault plane)... my experience with noisey correlation data makes me think this is hard to automate in a way that gives robust and reliable results (though it's easy to automate)... COSI-Corr has an option for this, which you can then go back an manualy adjust, which is quite a good way to go.
# 4. we need a way to automatically estiate the fault zone width... one option is to do this from the strain map, and automaticaly detect the fault-perpendicular width where the fault exceeds the elastic yield limit (0.5% strain). It's not necessarily the best way though, but it's one approach.
# 5. it would be nice to add a figure at the end which plots everything, i.e. the along-strike displacement, with the FZW added, etc , etc.
# 6. note... we could switch in other datasets, like the strain, and also topography, etc.


def create_shp(param, pixel_coords, name="shapefile", shp_type="None", nb_profile=0, nb_seg=0, folder='temp/'):
    transform = param
    
    temp_folder = folder + "/"
    """
    print(temp_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    """    
    shp_name = temp_folder + name + "_" + str(nb_seg) + ".shp"
    if shp_type == "point" :
        new_utm_coords = np.zeros((pixel_coords.shape[0], 2))
        
    
        for i, (x, y) in enumerate(pixel_coords):
            # Convert to pixel coordinates
            utm_x, utm_y = transform * (x, y)  # Apply the transform
            new_utm_coords[i] = (utm_x, utm_y)
    
    
        with shapefile.Writer(shp_name, shapeType=shapefile.POINT) as shp:
            shp.field("ID", "N")  # Add an integer field : nb profile
            shp.field("Num_seg", "N") # Add an integer field : nb fault segment
    
            for i, coord in enumerate(new_utm_coords):
                shp.record(nb_profile+i, nb_seg)  # Assign an ID to each point
                shp.point(coord[0], coord[1])  # Save (X, Y) as a point
                
    
    if shp_type == "line":  
        new_utm_coords = np.zeros((pixel_coords.shape[0], 2, 2))  # Each row has 2 points (X, Y)
    
        for i, (x1, y1, x2, y2) in enumerate(pixel_coords):  
            # Convert both points to UTM
            utm_x1, utm_y1 = transform * (x1, y1)  
            utm_x2, utm_y2 = transform * (x2, y2)  
            new_utm_coords[i] = [(utm_x1, utm_y1), (utm_x2, utm_y2)]
    
        # Create a shapefile with LineString geometries
        with shapefile.Writer(shp_name, shapeType=shapefile.POLYLINE) as shp:
            shp.field("ID", "N")  # Add an attribute field
            shp.field("Num_seg", "N") # Add an integer field : nb fault segment
        
            for i, coords in enumerate(new_utm_coords):
                shp.record(nb_profile+i, nb_seg)  # Assign an ID
                shp.line([coords.tolist()])  # Convert NumPy array to list and save as a LineString        
        
    # Create .prj file with EPSG:32611 (WGS 84 / UTM Zone 11N)
    with open(shp_name[:-4] + ".prj", "w") as prj:
        prj.write('''PROJCS["WGS 84 / UTM zone 11N",
            GEOGCS["WGS 84",
            DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
            PROJECTION["Transverse_Mercator"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",-117],
            PARAMETER["scale_factor",0.9996],
            PARAMETER["false_easting",500000],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
            AUTHORITY["EPSG","32611"]]''')


def remove_outliers_iqr_windowed(data, window=5, q_low=0.25, q_high=0.75, iqr_scale=1.5, infill=np.nan):
    data = np.asarray(data)
    result = np.copy(data)

    half_window = window // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)

        window_data = data[start:end]
        window_data = window_data[~np.isnan(window_data)]

        if len(window_data) < 3:
            continue  # Not enough data to calculate statistics

        val = data[i]
        if np.isnan(val):
            continue  # Skip NaNs

        q1 = np.quantile(window_data, q_low)
        q3 = np.quantile(window_data, q_high)
        iqr = q3 - q1
        lower = q1 - iqr_scale * iqr
        upper = q3 + iqr_scale * iqr

        if val < lower or val > upper:
            result[i] = infill

    return result


def robust_subset(data, threshold=2.5, min_points=5, fallback='median'):
    """
    Find indices of a robust subset of data based on deviation from a central value.
    
    Parameters:
    -----------
    data : array-like
        Input 1D array, possibly with NaNs.
    threshold : float
        Sigma multiplier for inlier detection (default 2.5).
    min_points : int
        Minimum number of points to keep. Falls back if fewer found.
    fallback : str
        'median' or 'mean' — determines fallback center estimate.
    
    Returns:
    --------
    inlier_indices : np.ndarray
        Indices (relative to original data) of inlier points.
    center : float
        Central value (median or mean) used.
    std_est : float
        Estimated standard deviation of inliers.
    """
    data = np.asarray(data)
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    
    if valid_data.size == 0:
        return np.array([], dtype=int), np.nan, np.nan

    center = np.nanmedian(valid_data)
    abs_dev = np.abs(valid_data - center)
    mad = np.nanmedian(abs_dev)
    std_est = mad * 1.4826  # Convert MAD to std estimate

    if std_est == 0 or np.isnan(std_est):
        std_est = np.nanstd(valid_data)

    inliers = abs_dev < threshold * std_est

    if np.sum(inliers) < min_points:
        # fallback: keep all valid points
        inliers = np.ones_like(valid_data, dtype=bool)
        if fallback == 'mean':
            center = np.nanmean(valid_data)
        else:
            center = np.nanmedian(valid_data)
        std_est = np.nanstd(valid_data)

    # Map inliers back to original indices
    valid_indices = np.flatnonzero(valid_mask)
    inlier_indices = valid_indices[inliers]

    return inlier_indices, center, std_est


def remove_marginal_outliers(tmp1, buffer=10, error_quantile=0.8):
    # buffer = margin around center where we won't remove outliers
    tmp1_tmp = tmp1*1.
    tmp1_tmp[:,plen-buffer:plen+buffer] = np.nan
    tmp1_stack = np.tile(np.nanstd(tmp1_tmp,axis=0), (tmp1.shape[0],1))
    tmp1_stack_diff_mean = np.nanquantile(np.abs(tmp1-tmp1_stack), error_quantile)
    tmp1_stack_mask = (tmp1 - tmp1_stack ) < tmp1_stack_diff_mean # 0.001
    tmp1_stack_mask[:,plen-buffer:plen+buffer] = 1.
    tmp1_masked = np.where(tmp1_stack_mask, tmp1, np.nan)
    #
    return tmp1_masked


def keep_if_majority_within_threshold(values, threshold):
    # Example usage:
    # threshold = 5.0  # adjust as needed
    # filter_func = partial(keep_if_majority_within_threshold, threshold=threshold)
    #
    # # Apply to 2D image
    # filtered = generic_filter(image, function=filter_func, size=5, mode='mirror')
    #
    center = values[len(values) // 2]
    neighbors = np.delete(values, len(values) // 2)  # remove center
    diffs = np.abs(neighbors - center)
    within_thresh = np.sum(diffs <= threshold)
    if within_thresh > len(neighbors) / 2:
        return center
    else:
        return np.nan
    
def keep_delta_or_majority_match(values, threshold):
    # threshold = 5.0
    # filter_func = partial(keep_delta_or_majority_match, threshold=threshold)
    #
    # filtered = generic_filter(image, function=filter_func, size=5, mode='mirror')
    #
    center = values[len(values) // 2]
    neighbors = np.delete(values, len(values) // 2)
    
    # Check how many neighbors are within threshold of the center
    diffs = np.abs(neighbors - center)
    close_to_center = np.sum(diffs <= threshold)
    
    # Also check if the neighborhood itself is flat
    neighbor_std = np.std(neighbors)
    
    if close_to_center > len(neighbors) / 2:
        # Majority match → keep
        return center
    elif neighbor_std <= threshold:
        # Spike on flat background → keep
        return center
    else:
        # Noisy or inconsistent region → remove
        return np.nan
    
    
def generalized_sigmoid_with_sign(x, x0, k1, k2, a_left, a_right, y0, gain):
    """
    Generalized sigmoid allowing both S and Z shapes:
    - gain: overall multiplier (positive or negative)
    """
    left = a_left * (x - x0)
    right = a_right * (x - x0)
    sigmoid = np.where(x < x0,
                       1 / (1 + np.exp(-k1*(x-x0))),
                       1 / (1 + np.exp(-k2*(x-x0))))
    return y0 + gain * (sigmoid + np.where(x < x0, left, right))


def bearing_to_xy(displacement, bearing_deg):
    bearing_rad = np.deg2rad(bearing_deg)
    dx = displacement * np.sin(bearing_rad)  # east
    dy = displacement * np.cos(bearing_rad)  # north
    return dx, -dy  # negate dy because positive is down for np arrays


def gaussian_weighted_mean(data, sigma=None, axis=-1):
    """
    Compute a Gaussian weighted mean along the specified axis, ignoring NaN values, with a Gaussian kernel centered on the middle of the vector.
    
    Parameters:
    - data: 2D array-like (n_rows x n_cols)
    - sigma: standard deviation of the Gaussian kernel. If None, defaults to the size of the data along the specified axis.
    - axis: The axis along which to compute the weighted mean. Default is -1 (the last axis).
    
    Returns:
    - weighted_mean: single value or array of Gaussian-weighted means, depending on the axis
    """
    data = np.asarray(data)
    
    # Get the size of the array along the specified axis
    n = data.shape[axis]
    
    # Default sigma is the size of the data along the specified axis if not provided
    if sigma is None:
        sigma = n
    
    # Create a Gaussian kernel centered at the middle of the array
    kernel = np.exp(-0.5 * (np.linspace(-n//2, n//2, n) / sigma)**2)
    kernel /= kernel.sum()  # Normalize the kernel
    
    # If axis is 0 (columns), apply the kernel along axis 0 (across rows)
    if axis == 0:
        # Apply the Gaussian filter along each column (axis=0), handling NaNs
        weighted_mean = np.apply_along_axis(
            lambda x: np.nansum(x * kernel) / np.nansum(np.where(np.isnan(x), 0, kernel)), 
            axis=0, arr=data
        )
    
    return weighted_mean



def smoothed_boxcar(plen, support_width, gaussian_sigma):
    """
    Create a boxcar smoothed by Gaussian convolution.

    plen: center index
    support_width: width of the box before smoothing
    gaussian_sigma: standard deviation of Gaussian (controls taper)
    """
    xx = np.arange(2*plen + 1)
    box = np.zeros_like(xx, dtype=float)

    center = plen
    half_width = support_width // 2

    box[(xx >= center - half_width) & (xx <= center + half_width)] = 1.0

    # Smooth edges
    smooth_box = gaussian_filter1d(box, sigma=gaussian_sigma)

    # Optional: normalize to maximum of 1
    smooth_box /= np.max(smooth_box)

    return smooth_box


# Define your known smoothing operation
def smooth(x):  # boxcar:11 gaussian:1.5 > CC 16x16 effective window
    x_smooth = uniform_filter1d(x, size=3, mode='nearest', origin=0)  # even sizes there is no true center, so the output is shifted 0.5 pixels
    x_smooth = gaussian_filter1d(x_smooth, sigma=0.375) #0.75)
    return x_smooth

# Weighted TV regularization
def weighted_tv(x, weights):
    return np.sum(weights * np.abs(np.diff(x)))

# Combined objective function
def objective(x, y_obs, weights, lam=0.1):
    return np.sum((smooth(x) - y_obs)**2) + lam * weighted_tv(x, weights)


def nanmedian_filter(vec, width):
    pad = width // 2
    padded = np.pad(vec, (pad, pad), mode='edge')
    result = np.empty_like(vec)

    for i in range(len(vec)):
        window = padded[i:i+width]
        result[i] = np.nanmedian(window)

    return result


def replace_outliers_robust(vec, window=5, threshold=5.0):
    vec = vec.copy()
    half = window // 2
    n = len(vec)

    for i in range(n):
        # Define window bounds
        start = max(0, i - half)
        end = min(n, i + half + 1)

        # Exclude the center value
        window_vals = np.delete(vec[start:end], i - start)
        # Remove NaNs
        window_vals = window_vals[~np.isnan(window_vals)]

        if len(window_vals) == 0 or np.isnan(vec[i]):
            continue  # can't replace if nothing to average with

        local_mean = np.nanmean(window_vals)
        if abs(vec[i] - local_mean) > threshold:
            vec[i] = local_mean

    return vec


def mad(arr, axis=None):
    arr = np.asarray(arr)
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=axis)
    return mad


def polyline_envelope(xtyt, distance):
    line = LineString(xtyt)  # Create polyline
    left_side, right_side = [], []

    coords = np.array(line.coords)  # Convert to array
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]

        # Compute direction vector
        dx, dy = p2 - p1
        length = np.hypot(dx, dy)

        # Compute unit perpendicular vector (-dy, dx)
        nx, ny = -dy / length, dx / length

        # Extend points perpendicular to the segment
        left_side.append((p1[0] + distance * nx, p1[1] + distance * ny))
        left_side.append((p2[0] + distance * nx, p2[1] + distance * ny))
        right_side.append((p1[0] - distance * nx, p1[1] - distance * ny))
        right_side.append((p2[0] - distance * nx, p2[1] - distance * ny))

    # Reverse right side so the envelope is a closed polygon
    envelope_coords = left_side + right_side[::-1]
    envelope_coords.append(envelope_coords[0])  # Close the polygon

    return Polygon(envelope_coords)


def rasterize_fault(vertices_array):
    # interpolate points onto the grid of a geotif
    # utm_coords = vertices_array[0:-1,0:2]
    utm_coords = vertices_array[:,0:2]

    # Ensure utm_coords is in the correct shape
    if utm_coords.ndim == 1:
        utm_coords = utm_coords.reshape(-1, 2)
    #np.save("utm_coords.npy", utm_coords)
    # Load the GeoTIFF to get pixel resolution
    with rasterio.open(ew_in) as src:
        pixel_size = src.res[0]  # Assuming square pixels, so one dimension is sufficient
        transform = src.transform

    # Step 1: Calculate cumulative distances along the line
    distances = np.sqrt(np.sum(np.diff(utm_coords, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Step 2: Define the interpolation functions
    unique_indices = np.unique(cumulative_distances, return_index=True)[1]
    cumulative_distances = cumulative_distances[unique_indices]
    utm_coords = utm_coords[unique_indices]

    x_interp = interp1d(cumulative_distances, utm_coords[:, 0], kind='cubic') # cubic for realfault, otherwise linear for synthetic!
    y_interp = interp1d(cumulative_distances, utm_coords[:, 1], kind='cubic') # cubic for realfault, otherwise linear for synthetic!

    # Step 3: Create an array of new distances based on the pixel size
    num_points = int(cumulative_distances[-1] / pixel_size)
    new_distances = np.linspace(0, cumulative_distances[-1], num_points+1)

    # Step 4: Interpolate the coordinates at new distances
    x_fine = x_interp(new_distances)
    y_fine = y_interp(new_distances)

    # Step 5: Combine the interpolated coordinates into a numpy array
    smoothed_coords = np.vstack((x_fine, y_fine)).T

    # Step 6: Convert UTM coordinates to pixel coordinates
    # Using the inverse of the affine transformation
    pixel_coords = np.zeros_like(smoothed_coords)
    for i, (x, y) in enumerate(smoothed_coords):
        # Convert to pixel coordinates
        pixel_x, pixel_y = ~transform * (x, y)  # Apply the inverse transform
        pixel_coords[i] = (pixel_x, pixel_y)

    fault = LineString(pixel_coords) # trace is x then y

    # Création du writer shapefile
    import shapefile
    w = shapefile.Writer("fault_line", shapeType=shapefile.POLYLINE)
    w.field("id", "N")  # Un champ numérique, par exemple
    w.line([list(fault.coords)])
    w.record(1)  # Enregistrement des attributs (correspond à la géométrie)
    w.close()
    return fault


def finite_strain(ew, ns, xres, yres, s=0, k=1, component=['exy','dilatation']):  #s = angle of fault
    no_components = len(component)
    # print(component)
    # print(f"rotation: {s} and smoothing: {k}")
    # Extract spatial metadata
    # input_transform  = [1,1,1,1,1,1]
    xres2=xres # resolution used for displacement gradient calc (may be modified later, if we dowsample)
    yres2=yres
    #    
    # downsample displacement field for strain calc (can also oversmooth if step<k)
    # k = 1 # smoothing kernel
    step = 1 # sliding window step (if step < k, we are oversampling) 
    # kernel = disk(k).astype(float)
    # kernel /= kernel.sum()  # Normalize to preserve mean
    # ew
    ew[np.isnan(ew)] = 0
    if k!=1 and k!=0:
        # ewsm = scipy.ndimage.uniform_filter(ew, size=k, mode='wrap') # average smoothing
        ewsm = gaussian_filter(ew, sigma=k, mode='wrap') # gaussian smoothing
        # ewsm = ndi.convolve(ew, kernel, mode='mirror') # disk smoothing
        # ewsm = ew
        # print("smoothing ew!")
    else:
        ewsm = ew
        # print("not smoothing ew!")
    ew2 = ewsm[0:-1:step,0:-1:step]
    ew2[ew2==0] = np.nan
    # ns
    ns[np.isnan(ns)] = 0
    if k!=1 and k!=0:
        # nssm = scipy.ndimage.uniform_filter(ns, size=k, mode='wrap') # average smoothing
        nssm = gaussian_filter(ns, sigma=k, mode='wrap') # gaussian smoothing
        # nssm = ndi.convolve(ns, kernel, mode='mirror') # disk smoothing
        # nssm = ns
        # print("smoothing ns!")
    else:
        nssm = ns
        # print("not smoothing ns!")
    ns2 = nssm[0:-1:step,0:-1:step]
    ns2[ns2==0] = np.nan

    ##
    # extract displacement gradient tensor
    # In a NumPy array system (y increases downward), using -yres2 in np.gradient makes the computed vorticity behave as if y were increasing upward again, restoring the standard interpretation: Positive vorticity = counterclockwise rotation  /  Negative vorticity = clockwise rotation
    dudy, dudx = np.gradient(ew2, yres2, xres2, edge_order=2)  
    dvdy, dvdx = np.gradient(ns2, yres2, xres2, edge_order=2)  #y decreases negatively down the array
    # 2nd and 3rd entires refer to the x and y pixel size (can also give as the x and y linspace coordinates)
    #
    F = np.array([[dudx, dudy], [dvdx, dvdy]])  # displacement gradient tensor
    
    # extract Finite strain (Green-Cauchy)   (needs identity matrix!)
    E11 = 0.5 * (2*dudx + dudx**2 + dvdx**2)
    E12 = 0.5 * (dudy + dvdx + dudx*dudy + dvdx*dvdy)
    E21 = 0.5 * (dudy + dvdx + dudx*dudy + dvdx*dvdy)
    E22 = 0.5 * (2*dvdy + dudy**2 + dvdy**2)

    E = np.array([[E11, E12],
                  [E21, E22]])
    
    # remove NaNs
    msk = np.isnan(E)
    E[msk] = 0
    F[msk] = 0
    
    # rotate basis from ew/ns to fault_parallel-fault_normal...
    if s!=0:
        
        sr = np.deg2rad(90-s) # angle from East to fault strike... converts from bearing from north (positive CW)... s-90, because north is negative
        # sr = np.deg2rad(-s)  # angle from East to fault strike
        R = np.array([[np.cos(sr), -np.sin(sr)],
                      [np.sin(sr),  np.cos(sr)]])  # CCW rotation matrix
        E = np.moveaxis(E, [0, 1], [-2, -1])
        # Rotate each 2x2 tensor in the array E (shape: nx x ny x 2 x 2)
        Er = np.einsum('ab,...bc,cd->...ad', R, E, R.T)
        # Optional: preserve NaNs
        # Er[np.isnan(E)] = np.nan
        mask = np.isnan(E).any(axis=(0, 1))
        Er[..., mask] = np.nan
        #
        Er = np.moveaxis(Er, [-2, -1], [0, 1])  # Back to (2, 2, 1294, 1377)
        E = Er
    

    strain_full_out = np.zeros((dudy.shape[0],dudy.shape[1], no_components))
    ff = -1
    for f in component:
        ff += 1
        # print(f)
        if f=='dudy':
            strain_out = dudy
        if f=='dudx':
            strain_out = dudx
        if f=='dvdy':
            strain_out = dvdy
        if f=='dvdx':
            strain_out = dvdx
            
        if f=='vorticity':
            # extract vorticity
            strain_out = np.rad2deg(0.5*(F[0,1]-F[1,0]))  # rotational component of strain tensor ... negative = R-L S-S
            # NOT the rotation tensor! (which is more complex to solve for)
        
        if f=='exx':
            # normal
            strain_out = E[0,0] # exx
        if f=='eyy':
            strain_out = E[1,1] # eyy
        if f=='exy':
            # print("extracting exy")
            strain_out = (E[0,1]+E[1,0])/2 # same as E[1,0], or (dudy + dvdx) / 2
            # exy = (dudy + dvdx) / 2 # shear component in xy, same as E[0,1] or E[1,0]
        
        # magTotalShear = (E[0,1]**2 + 0.25*(E[0,0] - E[1,1])**2)**0.5 # magnitude of the TOTAL shear strain 
        # magShear = (E[0,1]**2 + E[1,0]**2)**0.5 # mag of shear strain tensor on a specific plane defined by the principal axes, inclined at theta (NOT max shear strain)... mag of off-diagonal components of strain tensor
        if f=='dilatation':
            # print("extracting dilatation")
            strain_out  = np.trace(E[:,:]) # dilatation... same as E[0,0]+E[1,1] or F[0,0] + F[1,1]
        #
        if f=='maxShear':
            # maxShear = ((E[0,0] - E[1,1])**2 + 4*E[0,1]**2)**0.5  # same as 2nd-inv... definition: max shear strain on any plane... J2 Von Mise strain    
            strain_out = 0.5 * ((E[0,0] - E[1,1])**2 + 4*E[0,1]**2)**0.5
        
        if f=='compression':
            strain_out = (E[0,0]+E[1,1])/2  #(average of the normal strain components)
        
        strain_full_out[:,:,ff] = strain_out

    return strain_full_out


t0 = time.time()
# TO MODIFY
""" 
# images
ew_in=sys.argv[1][:] #"../corr/disp_ew_FILL_TVL1_smooth_2.0_mosaic_detrended.tif" #"full/disp_ew_FILL_TVL1_smooth_2.0_mosaic_detrended_poly2.tif"
ns_in=sys.argv[2][:] #"../corr/disp_ns_FILL_TVL1_smooth_2.0_mosaic_detrended.tif" #"full/disp_ns_FILL_TVL1_smooth_2.0_mosaic_detrended_poly2.tif"
# dem_in=sys.argv[3][:] #"../../aw3d30/sagaing_nutm46-adj_40m.tif"
#strain1_in=sys.argv[3][:]
#strain2_in=sys.argv[4][:]
rupture=sys.argv[3][:]
cores=int(sys.argv[4][:])
"""
"""
ew_in = "/data/cycle/rocamori/NAPP/3Dcorrelation/hm_all_area/1994_2002_ew.tif"
ns_in = "/data/cycle/rocamori/NAPP/3Dcorrelation/hm_all_area/1994_2002_ns.tif"
rupture = "/data/cycle/rocamori/NAPP/shapefile/ruptures_hm_main_simplified_V2.shp"
cores = int(sys.argv[1][:])

ew_in = "/data/cycle/rocamori/NAPP/3Dcorrelation/landers_all_area/1989_1994_ew.tif"
ns_in = "/data/cycle/rocamori/NAPP/3Dcorrelation/landers_all_area/1989_1994_ns.tif"
rupture = "/data/cycle/rocamori/NAPP/shapefile/ruptures_landers_main_simplified_V2.shp"
cores = int(sys.argv[1][:])
"""

site = "landers"
ew_in = "/data/cycle/rocamori/Derivatives_Hector_Mine/resampled_1m/"+site+"_ew_1m.tif"
ns_in = "/data/cycle/rocamori/Derivatives_Hector_Mine/resampled_1m/"+site+"_ns_1m.tif"
rupture = "/data/cycle/rocamori/NAPP/shapefile/ruptures_"+site+"_main_simplified_V2.shp"
cores = int(sys.argv[1][:])
"""
ew_in=sys.argv[1][:] 
ns_in=sys.argv[2][:] 
rupture=sys.argv[3][:]
cores=int(sys.argv[4][:])
"""
# /!\ values in pixels
# TO MODIFY
background_limit = [-750, 750]
near_field_limit = [10, 60]
far_field_limit = [near_field_limit[1], near_field_limit[1]+50]
stdthr = 3

# ew_in="EW_Ridgecrest_1m_small_area.tif"
# ns_in="NS_Ridgecrest_1m_small_area.tif"
# # # dem_in="/data/projects/optical/hollingj/myanmar/sagaing/aw3d30/sagaing_nutm46-adj.tif"
# # # strain1_in="exy.tif"
# # # strain2_in="dilatation.tif"
# rupture="new_fault.shp"
# cores = 2

# import a shapefile to get the utm coordinates of the vertices
# Load the line shapefile
#gdf = gpd.read_file(rupture)  #"../shp/rupture2.shp")

# Check and set the original CRS if needed, assuming WGS84 (EPSG:4326) here
# if gdf.crs is None:
#     gdf.set_crs("EPSG:4326", inplace=True)

# Reproject to UTM (replace with the appropriate UTM zone, e.g., EPSG:32611 for Zone 11N)
#utm_gdf = gdf

# # Step 3: Extract vertices and store them in a numpy array
all_vertices = []

# Loop through each geometry in the GeoDataFrame
"""
for geom in utm_gdf.geometry:
    if isinstance(geom, LineString):
        # Extract coordinates from LineString
        coords = np.array(geom.coords)
        all_vertices.append(coords)
    
    elif geom.geom_type == 'MultiLineString':
        # Handle MultiLineString by iterating over each LineString part
        for part in geom:
            coords = np.array(part.coords)
            all_vertices.append(coords)
"""
# Extract vertices
fault_sf = shapefile.Reader(rupture)
fault_shapes = fault_sf.shapes() 

num_folder = 1
for shape in fault_shapes :
    
    temp_folder = f"seg_{num_folder:03d}"
    print(temp_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    # Extract vertices
    all_vertices = []
    if len(shape.points) > 1:  # Ensure it's a valid line geometry
        line = LineString(shape.points)
        all_vertices.append(np.array(line.coords))
    else :
        print("error, it's not a line (the number of points is ≤ 1)")
        pass
        sys.exit()
     
    # If the shapefile contains MultiLineStrings, handle them
    if any(isinstance(geom, MultiLineString) for geom in all_vertices):
        processed_vertices = []
        for geom in all_vertices:
            if isinstance(geom, MultiLineString):
                for part in geom:
                    processed_vertices.append(np.array(part.coords))
            else:
                processed_vertices.append(geom)
        all_vertices = processed_vertices
     
    
    # Combine all vertices into a single numpy array
    vertices_array = np.vstack(all_vertices)
    print("Extracted vertices in UTM coordinates")
    # print(vertices_array)
    
    fault = rasterize_fault(vertices_array)
    #np.save("fault_smoothed.npy", fault)
    print("Fault saved")
    
    # NOTE... currently, flipping the NS to S-positive seems to be ok in terms of retrieving good fault-parallel and normal slip, and strain
    
    # load displacements
    # Open the GeoTIFF
    with rasterio.open(ew_in) as dataset:
        disp_ew_fill = dataset.read(1)
        profile = dataset.profile
        minx, miny, maxx, maxy = dataset.bounds
        transform = dataset.transform  # Affine transformation (to convert pixel coords to utm, use: transform * (px, py))
    with rasterio.open(ns_in) as dataset:
        disp_ns_fill = -dataset.read(1)
    xres=transform[0] #40
    yres=transform[0] #40
    
    # fault profiles
    x, y = np.array(fault.xy)
    
    # smooth fault line (but you loose the boundary points)
    smk = 15 # smoothing kernel
    kernel = np.ones(smk) / smk
    x = np.convolve(x, kernel, mode='valid')
    y = np.convolve(y, kernel, mode='valid')
    
    span = 1
    angle = np.zeros(x.shape)
    # plt.figure(dpi=1200); plt.scatter(x,y, marker='s', s=0.1, color='None', facecolors='black'); plt.gca().set_aspect('equal'); plt.gca().invert_yaxis()
    
    # calculate local fault angle
    for i in range(span,x.shape[0]-span):
        if i < span:
            x0 = np.mean(x[:i])
            y0 = np.mean(y[:i])
        elif i > x.shape[0]-span:
            x0 = np.mean(x[i:])
            y0 = np.mean(y[i:])
        else:
            x0 = np.mean(x[i-span:i])
            x1 = np.mean(x[i:i+span])
            y0 = np.mean(y[i-span:i])
            y1 = np.mean(y[i:i+span])
            xmean = (x0+x1)/2
            ymean = (y0+y1)/2
        dx = x1-x0
        dy = y1-y0
        azimuth = (90 + np.rad2deg(np.arctan2(dy, dx)))%360 #angle relative to north
        angle[i] = azimuth #180 - (azimuth + 360) % 360  #minus 180 is because pixels increase down, therefore we need to mirror (if we're workng in meter/km then positive y is upwards, so we remove -180)
    angle = angle[span:-span]
    x0 = x[span:-span]
    y0 = y[span:-span]
    
    # plot along-fault strike angle
    # plt.figure(dpi=300); plt.plot(angle)
        
    
    # extract displacement profiles
    #%%
    ####################################################################### VARIABLES ###########################################################################################   
    # get end points of each profile
    x1 = np.zeros(x0.shape)
    y1 = np.zeros(y0.shape)
    x2 = np.zeros(x0.shape)
    y2 = np.zeros(y0.shape)
    plen = 500  # fault-perp profile 1/2 length  # attention pixels
    stack = 300  # half stack width
    
    # define a weighting function for reducing noise around peak strain (so we can better locate it)
    # Set expected peak position and spread
    # expected_peak = plen
    # xx = np.arange((2*plen+1))
    # sigma = plen * 0.03  # adjust depending on how wide your peak region is
    # w = np.exp(-0.5 * ((xx - expected_peak) / sigma)**2) # careful... we assume the fault is perfectly centered on zero, and it might not be, thus this could bias the result slightly (if the width of the gaussian is narrow relative to the peak strain spike)
    w = smoothed_boxcar(plen, 25, 1)
    wx = np.linspace(0,1,w.shape[0])
    wx2 = np.linspace(0,1,int(w.shape[0]*1e3))
    interp_func = interp1d(wx, w, kind='slinear')
    ww = interp_func(wx2)
    for i in range(x0.shape[0]):  # currently NOT optimized for the DEM stats or strain (redundant calculations!)
       theta_perp = np.deg2rad(angle[i] + 90)  # or -90
       dx = plen * np.sin(theta_perp)
       dy = plen * np.cos(theta_perp)
       x1[i] = x0[i] + dx
       y1[i] = y0[i] - dy  # If you're using image coordinates (origin at top-left), then Y increases downward, so flip the dy
       # y1[i] = y0[i] + dy
       x2[i] = x0[i] - dx
       y2[i] = y0[i] + dy  # If you're using image coordinates (origin at top-left), then Y increases downward, so flip the dy
       # y2[i] = y0[i] - dy
    
    # interpolate the displacement profiles
    for n in [2,5] : # [1,2,3,4,5]: #range(2,6):
       if n==2:
           # dxf_infill = strain1[:,:] # shear strain exy
           # dxf_smooth_infill = strain1_smooth[:,:] # shear strain exy smoothed! (better fault location)
           # dyf_infill = strain2[:,:] # dilatation
           dxf_infill = disp_ew_fill
           dyf_infill = disp_ns_fill
           dxf_infill[dxf_infill==0] = np.nan
           dyf_infill[dyf_infill==0] = np.nan
       if n==5:
           dxf_infill = disp_ew_fill
           dyf_infill = -disp_ns_fill  # switch displacement to +north (because we load as south-positive)
           dxf_infill[dxf_infill==0] = np.nan
           dyf_infill[dyf_infill==0] = np.nan
       # if n==6:
           # dxf_infill = dem
           # dyf_infill = dem
       #
       # extract displacements in parallel (for speed!)
       prof_store = np.zeros((x0.shape[0],4,int(2*plen)+1))
       store_length = prof_store.shape[0]
       prof_length = prof_store.shape[2]
       
       def extract_disp(i,n,store_length, prof_length):
           print(f"extracting displacements (and strain)... loop {n}, processing line {i} of {store_length}", flush=True)
           prof_store_tmp = np.zeros((4, prof_length))
           # for i in range(x0.shape[0]):
           # print(f"extracting displacements: {i} of {x0.shape[0]}")
           expwd = 5
           prof = np.linspace([x1[i],y1[i]], [x2[i],y2[i]], num=(2*plen)+1)
           # ew/ns displacements
           # prof_store[i,0,:] = map_coordinates(dxf_infill, np.fliplr(prof).T, order=1)
           # prof_store[i,1,:] = map_coordinates(dyf_infill, np.fliplr(prof).T, order=1)
           tmp1 = dxf_infill[int(np.min(prof[:,1])-expwd):int(np.max(prof[:,1])+expwd), int(np.min(prof[:,0])-expwd):int(np.max(prof[:,0])+expwd)]
           tmp2 = dyf_infill[int(np.min(prof[:,1])-expwd):int(np.max(prof[:,1])+expwd), int(np.min(prof[:,0])-expwd):int(np.max(prof[:,0])+expwd)]
           # # remove outliers (except for +/- 10 pixels around peak)
           # tmp1 = remove_marginal_outliers(tmp1,10,0.99)
           # tmp2 = remove_marginal_outliers(tmp2,10,0.99)
           if n==2:
               try:
                   strain_out = finite_strain(tmp1, tmp2, xres, yres, angle[i], k=1, component=['exy', 'dilatation', 'maxShear']) # exy = shear / exx = tension dir parallel to strike
                   tmp1 = np.abs(strain_out[:,:,0])
                   tmp2 = strain_out[:,:,1]
                   # tmp1_smooth = (denoise_tv_chambolle(tmp1,weight=3))
                   tmp1_smooth = strain_out[:,:,2]
                   # remove outliers (except for +/- 10 pixels around peak)
                   tmp1 = remove_marginal_outliers(tmp1,15,0.5)
                   tmp2 = remove_marginal_outliers(tmp2,15,0.5)
               except Exception as e:
                   tmp1 = np.full_like(tmp1, np.nan, dtype=float)
                   tmp2 = np.full_like(tmp1, np.nan, dtype=float)
                   tmp1_smooth = np.full_like(tmp1, np.nan, dtype=float)
               # # tmp1_smooth = denoise_tv_bregman(tmp1,weight=2, isotropic=True)
               # tmp1_smooth = dxf_smooth_infill[int(np.min(prof[:,1])-expwd):int(np.max(prof[:,1])+expwd), int(np.min(prof[:,0])-expwd):int(np.max(prof[:,0])+expwd)]
           # tmp3 = dxf_infill2[int(np.min(prof[:,1])-5):int(np.max(prof[:,1])+5), int(np.min(prof[:,0])-5):int(np.max(prof[:,0])+5)]
           # tmp4 = dyf_infill2[int(np.min(prof[:,1])-5):int(np.max(prof[:,1])+5), int(np.min(prof[:,0])-5):int(np.max(prof[:,0])+5)]
           tmp_prof = np.fliplr(prof).T
           tmp_prof[0,:] = tmp_prof[0,:] - int(np.min(prof[:,1])) +expwd  # y
           tmp_prof[1,:] = tmp_prof[1,:] - int(np.min(prof[:,0])) +expwd  # x
           prof_store_tmp[0,:] = map_coordinates(tmp1, tmp_prof, order=1)  # tmp_prof= y then x
           prof_store_tmp[1,:] = map_coordinates(tmp2, tmp_prof, order=1)
           if n==2:
               prof_store_tmp[2,:] = map_coordinates(tmp1_smooth, tmp_prof, order=1)
           # if n==2:
           #     prof_store_tmp[2,:] = map_coordinates(tmp3, tmp_prof, order=1)
           #     prof_store_tmp[3,:] = map_coordinates(tmp4, tmp_prof, order=1)
           # fault_normal/fault_parallel
           if n==5:
               theta = np.radians(90 - angle[i]) 
               tmp3 =  tmp1 * np.cos(theta) + tmp2 * np.sin(theta) # flt-parallel
               tmp4 = -tmp1 * np.sin(theta) + tmp2 * np.cos(theta)  # flt-normal
               prof_store_tmp[3,:] = map_coordinates(tmp3, tmp_prof, order=1)  # flt-parallel
               prof_store_tmp[2,:] = map_coordinates(tmp4, tmp_prof, order=1)  # flt.normal
                # tmp3 = (tmp1*np.cos(np.radians(angle[i]))) + (tmp2*np.sin(np.radians(angle[i])))
                # tmp4 = (-tmp1*np.sin(np.radians(angle[i]))) + (tmp2*np.cos(np.radians(angle[i])))
                # prof_store_tmp[3,:] = map_coordinates(tmp3, tmp_prof, order=1)  # flt-parallel
                # prof_store_tmp[2,:] = map_coordinates(tmp4, tmp_prof, order=1)  # flt.normal
                # # for E-positive and S-positive
                # prof_store_tmp[2,:] = ( prof_store_tmp[0,:] * np.cos(np.deg2rad(angle[i])) ) + \
                #                     ( -prof_store_tmp[1,:] * np.sin(np.deg2rad(angle[i])) )
                # prof_store_tmp[3,:] = (-prof_store_tmp[0,:] * np.sin(np.deg2rad(angle[i])) ) + \
                #                     ( -prof_store_tmp[1,:] * np.cos(np.deg2rad(angle[i])) )
                  # for E-positive and N-positive
                  # prof_store_tmp[2,:] = ( prof_store_tmp[0,:] * np.cos(np.deg2rad(angle[i]-90)) ) + \
                  #                     ( prof_store_tmp[1,:] * np.sin(np.deg2rad(angle[i]-90)) )   # fault-parallel (x is rotated to fault-par strike)
                  # prof_store_tmp[3,:] = (-prof_store_tmp[0,:] * np.sin(np.deg2rad(angle[i]-90)) ) + \
                  #                     ( prof_store_tmp[1,:] * np.cos(np.deg2rad(angle[i])-90) )  # fault-normal
             #
           return i, prof_store_tmp
       
        # parallelization
       with ProcessPoolExecutor(max_workers=cores) as executor:
           futures = [executor.submit(extract_disp, i, n, store_length, prof_length) for i in range(prof_store.shape[0])]
           for future in as_completed(futures):
               i, prof_store_tmp = future.result()
               prof_store[i,0,:] = prof_store_tmp[0,:] # shear  /  EW
               prof_store[i,1,:] = prof_store_tmp[1,:] # dilatation  /  NS
               prof_store[i,2,:] = prof_store_tmp[2,:] # smooth shear  /  flt-normal
               prof_store[i,3,:] = prof_store_tmp[3,:] # empty  /  flt-parallel
    
               
    
       # reshift profiles onto primary faut core (peak strain)... reduces artifacts when stacking along-strike... this is sub-pixel alignment!
       if n==2:
           store_shifts = np.zeros((prof_store.shape[0],1))
           for i in range(0,prof_store.shape[0]):
               shift_style = 'max shear strain'
               # toot = prof_store[i,0,:]
               # weighted_toot = toot * w
               # store_shifts[i] = np.argmax(weighted_toot)-plen
               print(f"shifting profile (non-stacked)...loop {n}, {i} of {prof_store.shape[0]}")
    
               ysp = prof_store[i,0,:] * w  # align on SMOOTHED shear strain (larger signal)
               xsp = np.linspace(0,ysp.shape[0]-1,ysp.shape[0])  # minus 1 is to account for the +1 shift right, so we remove half of that
               # xsp = np.linspace(0,ysp.shape[0],ysp.shape[0])-1.5  # minus 1 is to account for the +1 shift right, so we remove half of that
    
               # IF making synthetics... mask below to end of loop, as we don't need to locate the shifts (we know them already)
               xsp2 = xsp
               ysp2 = ysp
               # # just for synthetics (because the sharp changes cause problems with the spline fitting)
               # mask = np.where(np.abs(np.gradient(ysp))> 0.0005)[0]
               # # mask = np.unique(np.hstack((mask,mask-1,mask+1)))  # erode the positions of the bad points if needed
               # xsp2 = np.delete(xsp, mask)
               # ysp2 = np.delete(ysp, mask)
    
               valid = ~np.isnan(ysp)
               xsp2 = xsp2[valid]
               ysp2 = ysp2[valid]
    
               # raw
               # use a savgol filter to smooth low-res, then upsample with interp1?
               # yspf=savgol_filter(ysp, window_length=15, polyorder=5, mode='nearest')
               try:
                   spline = UnivariateSpline(xsp2, ysp2, s=1e-9)  # s is the smoothing factor    ... 1e-8 CC / 1e-3 for synthetics / 1e-8 for data
                   # synthetics
                   # spline = UnivariateSpline(xsp2, ysp2, s=6e-6)  # s is the smoothing factor    ... 1e-3 for synthetics / 1e-8 for data
                   # spline = PchipInterpolator(xsp2, ysp2)  # better at preserving the sharp edges... but not sure
        
                   # Define a fine range of x values for evaluation
                   xno = int(xsp.shape[0]*1e3)
                   x_fine = np.linspace(xsp.min(), xsp.max(), xno)
                   xdivfct = (xno/(xsp.max()-1-xsp.min()))
                   y_fine = spline(x_fine)
                   x_shift = ((np.nanargmax(y_fine)/xno)*(xsp.shape[0]))-(xsp.shape[0]/2) # profile now scaled between -plen to +plen thus even... 1/2 pixels are one side, 1/2 the other
                   # if (x_shift >= plen//3) or (x_shift <= -plen//3):
                   if (x_shift >= 8) or (x_shift <= -8): # I computed store_shifts without bounding, and computed the stdev = 4.9, thus +/- 10 should be a good constraint                  # TO MODIFY
                       x_shift = 0 # or np.nan
               except Exception as e:
                   x_shift = 0
               print(x_shift)
               # calculate dx/dy adjustment to x0/y0 form the shift and bearing
               # dx, dy = bearing_to_xy(x_shift,angle[i])
               store_shifts[i] = x_shift
    
       if n==2:
           np.save(f"{temp_folder}/prof_store_strain1c{num_folder:03d}.npy", prof_store)
           
           # interpolate zero shifts (which are where it's failed)
           store_valid = store_shifts!=0
           store_row = np.linspace(0,store_shifts.shape[0],store_shifts.shape[0]).reshape(-1,1)
           store_shifts_valid = store_shifts[store_valid]
           store_row_valid = store_row[store_valid]
           interp_func = interp1d(store_row_valid,  store_shifts_valid, kind='slinear', bounds_error=False, fill_value=np.nan)
           store_shifts = interp_func(store_row)
           #
           # store_shifts = median_filter(store_shifts, size=3, mode='nearest')
           # store_shifts = nanmedian_filter(store_shifts, width=30)
           # store_shifts = replace_outliers_robust(store_shifts, window=10, threshold=1.0)  # moving mean smoothing of shifts, to remove outliers
           # could think about removing values set to 0 and interpolating them based on their neighbors?
           tmpx = np.linspace(0,ysp.shape[0]-1,ysp.shape[0])
           # tmpx = np.linspace(0,ysp.shape[0]-1,ysp.shape[0])-plen  #np.linspace(0,1+(2*plen),1+(2*plen))
    #       if shift_style=='zero displacement':
    #           tmpx2 += 0.5
           for i in range(0,prof_store.shape[0]):
               tmpx2 = tmpx - store_shifts[i]
               for p in [0, 1, 2]: #, 2, 3]:
                   valid = ~np.isnan(prof_store[i,p,:])
                   try:
                       interp_func = interp1d(tmpx2[valid],  prof_store[i,p,:][valid], kind='slinear', bounds_error=False, fill_value=np.nan)
                       tmp = interp_func(tmpx)
                       tmp[tmp==0.0] = np.nan
                       prof_store[i,p,:] = tmp
                   except Exception as e:
                       prof_store[i,p,:] = np.full(tmpx.shape[0], np.nan)
                   print("strain:",i,p)
           np.save(f"{temp_folder}/prof_store_strain2c{num_folder:03d}.npy", prof_store)
    
       if n==5:
           store_shifts = replace_outliers_robust(store_shifts, window=10, threshold=1.0)
           tmpx = np.linspace(0,ysp.shape[0]-1,ysp.shape[0])  #np.linspace(0,1+(2*plen),1+(2*plen))
           for i in range(0,prof_store.shape[0]):
               tmpx2 = tmpx - store_shifts[i]
               for p in [0, 1, 2, 3]:
                   valid = np.isfinite(prof_store[i, p, :]) & np.isfinite(tmpx2)
                   try:
                       interp_func = interp1d(tmpx2[valid], prof_store[i, p, :][valid],
                                              kind='slinear', bounds_error=False, fill_value=np.nan)
                       prof_store[i, p, :] = interp_func(tmpx)
                   except Exception as e:
                       prof_store[i, p, :] = np.full(tmpx.shape[0], np.nan)
                   print("displacements:",i,p)
           np.save(f"{temp_folder}/prof_store_disp1c{num_folder:03d}.npy", prof_store)

    
       if n==6:
           tmpx = np.linspace(0,1+(2*plen)-1,1+(2*plen))
           for i in range(0,prof_store.shape[0]):
               for p in [0]:
                   valid = ~np.isnan(prof_store[i,p,:])
                   interp_func = interp1d((tmpx-store_shifts[i])[valid],  prof_store[i,p,:][valid], kind='slinear', bounds_error=False, fill_value=np.nan)
                   prof_store[i,p,:] = interp_func(tmpx)
                   print("dem:",i,p)
    
       # stack profiles  ...
       print("stacking profiles now... in parallel!")
       # stack = 1000  # half stack width
       prof_store_stats = np.zeros((prof_store.shape[0],8,int(2*plen)+1))  # update to 10 if x-shifting profs
       store_stats_length = prof_store_stats.shape[0]
       prof_stats_length = prof_store_stats.shape[2]
       # for i in range(stack,prof_store.shape[0]-stack):
       def stacking_profiles(i,n,store_length, prof_length):
           print(f"stacking profiles: loop {n}, processing line {i} of {store_length}", flush=True)
           # prof_store_stats_tmp = np.zeros((8,int(2*plen)+1))
           prof_store_stats_tmp = np.full((8, int(2*plen)+1), np.nan)
           # print(n,i,prof_store.shape[0]-(stack))
           if i < stack:
               x0 = np.mean(x[:i])  # could use np.median
               y0 = np.mean(y[:i])  # could use np.median
           elif i > x.shape[0]-stack:
               x0 = np.mean(x[i:])  # could use np.median
               y0 = np.mean(y[i:])  # could use np.median
           else:
               # stack ew / shear / dem
               track_nan = np.sum(np.isnan(prof_store[i-stack:i+stack,1,:]), axis=0) # no of nans per stack for each prof position
               if np.sum(track_nan<stack//3)>=plen//1.5:  # if the number nans < 1/3 stack length for more than 1/2 of plen, continue (else set to nan)
                   prof_store_stats_tmp[0,:] = np.nanmedian(prof_store[i-stack:i+stack,0,:], axis=0) # np.nanmean / np.nanmedian (SLOW) / gaussian_weighted_mean
                   prof_store_stats_tmp[1,:] = np.nanstd(prof_store[i-stack:i+stack,0,:], axis=0) # np.nanstd / mad
                   #
                   if n==2 or n==5: # stack dilatation / ns
                           prof_store_stats_tmp[2,:] = np.nanmedian(prof_store[i-stack:i+stack,1,:], axis=0)
                           prof_store_stats_tmp[3,:] = np.nanstd(prof_store[i-stack:i+stack,1,:], axis=0)
                       #
                           prof_store_stats_tmp[4,:] = np.nanmedian(prof_store[i-stack:i+stack,2,:], axis=0)  # smoothed shear
                           prof_store_stats_tmp[5,:] = np.nanstd(prof_store[i-stack:i+stack,2,:], axis=0)
                   if  n==5: # stack fault normal/parallel # add n==2 if you want to recover dudx / dvdy
                       # prof_store_stats_tmp[4,:] = np.nanmean(prof_store[i-stack:i+stack,2,:], axis=0)
                       # prof_store_stats_tmp[5,:] = np.nanstd(prof_store[i-stack:i+stack,2,:], axis=0)
                       #
                       prof_store_stats_tmp[6,:] = np.nanmedian(prof_store[i-stack:i+stack,3,:], axis=0)
                       prof_store_stats_tmp[7,:] = np.nanstd(prof_store[i-stack:i+stack,3,:], axis=0)
               #
               return i, prof_store_stats_tmp
       # parallelization for stacking!
       with ProcessPoolExecutor(max_workers=cores) as executor:
           futures = [executor.submit(stacking_profiles, i, n, store_stats_length, prof_stats_length) for i in range(stack,prof_store.shape[0]-stack)]
           for future in as_completed(futures):
               i_idx, prof_store_stats_tmp = future.result()
               prof_store_stats[i_idx,0,:] = prof_store_stats_tmp[0,:]
               prof_store_stats[i_idx,1,:] = prof_store_stats_tmp[1,:]
               prof_store_stats[i_idx,2,:] = prof_store_stats_tmp[2,:]
               prof_store_stats[i_idx,3,:] = prof_store_stats_tmp[3,:]
               prof_store_stats[i_idx,4,:] = prof_store_stats_tmp[4,:]
               prof_store_stats[i_idx,5,:] = prof_store_stats_tmp[5,:]
               prof_store_stats[i_idx,6,:] = prof_store_stats_tmp[6,:]
               prof_store_stats[i_idx,7,:] = prof_store_stats_tmp[7,:]
       
        # trim smoothing boundaries
       prof_store_stats = prof_store_stats[stack:-stack,:,:]
       fault_pos = np.column_stack((x0,x1,x2,y0,y1,y2,angle))  # x0,y0 = on-fault, x1,y1=left, x2,y2=right
       fault_pos = fault_pos[stack:-stack,:]
       prof_dist_pxls = np.linspace(-plen, plen, num=(2*plen)+1)
       if n==2:
           prof_store_stats_strain = prof_store_stats
       if n==5:
           prof_store_stats_disp = prof_store_stats
       if n==6:
           prof_store_stats_dem = prof_store_stats
    
    
    # fault_pos = np.column_stack((x0,x1,x2,y0,y1,y2,angle))  # x0,y0 = on-fault, x1,y1=left, x2,y2=right
    # fault_pos = fault_pos[stack:-stack,:]
    # prof_dist_pxls = np.linspace(-plen, plen, num=(2*plen)+1)
    print("saving initial results")
    np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_strainc{num_folder:03d}.npy", prof_store_stats_strain)
    np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_dispc{num_folder:03d}.npy", prof_store_stats_disp)
    np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_store_shiftsc{num_folder:03d}.npy", store_shifts)

    
    store_shifts[np.isnan(store_shifts)] = 0
    store_shifts2 = np.zeros_like(store_shifts)
    # for ff in range(prof_store_stats_strain.shape[0]):  # basic loop... see below for parallelized
    def final_shift_estimate(ff, total_size):
        # TODO
        # update x0,y0 (and x1y1,x2y2) to incororate new shift, so we keep track of new fault position
        # print(f"processing line {ff} of {prof_store_stats_strain.shape[0]}")
        # re_shift = np.zeros((3, prof_store_stats_strain.shape[2]))
     
        prof = ff   # prof = 2700 is where is collapses
        ysp = prof_store_stats_strain[prof,0,:]  # 0 = shear / 2 = dilatation / 4 = smoothed shear  !!! careful about sign! if r-l negate!
        # yspd = prof_store_stats_strain[prof,2,:]  # 0 = shear / 2 = dilatation
        # yspss = prof_store_stats_disp[prof,6,:]  # 0 = shear / 2 = dilatation
        xsp = prof_dist_pxls  # because the profiles has been shifted so the peak is a 1 (not zero)... at least, that's what I see in the syntethics!
        w = smoothed_boxcar(plen, 8, 1) # (len, support_width, gaussian_sigma)
        #
        xsp2 = xsp
        ysp2 = ysp    
        # yspd2 = yspd
        # yspss2 = yspss
        #
        # valid = (np.isnan(ysp2) + np.isnan(yspd2) + np.isnan(yspss2)) < 1
        valid = (np.isnan(ysp2)) < 1
        xsp2 = xsp2[valid]
        ysp2 = ysp2[valid]
        # yspd2 = yspd2[valid]
        # yspss2 = yspss2[valid]
        w2 = w[valid]
        #
        # ysp2 = gaussian_filter1d(ysp2, sigma=(plen/2))
        try:
            spline = UnivariateSpline(xsp2, (ysp2)*w2, s=1e-9)  # s is the smoothing factor
            #
            # Define a fine range of x values for evaluation
            xno = int(xsp.shape[0]*1e3)
            x_fine = np.linspace(xsp.min(), xsp.max(), xno)
            y_fine = spline(x_fine)
            # compute remaining shift
            x_shift2 = (np.nanargmax(y_fine)/xno)*xsp.shape[0] - (xsp.shape[0]/2) # minus 0.5, because the peak is interpolated between 0 and 1
            if (x_shift2 >= 4) or (x_shift2 <= -4): # I computed store_shifts without bounding, and computed the stdev = 4.9, thus +/- 10 should be a good constraint
                x_shift2 = 0 # or np.nan
                print(ff)
        except Exception as e:
            x_shift2 = 0
        print(f"final re-shift correction... row: {ff} (of {total_size}), profile shift: {x_shift2}")
        return ff, x_shift2
    
    # parallelization
    with ProcessPoolExecutor(max_workers=cores) as executor1:
       futures1 = [executor1.submit(final_shift_estimate, ff, prof_store_stats_disp.shape[0]) for ff in range(prof_store_stats_strain.shape[0])]
       for future in as_completed(futures1):
           ff_idx, x_shift2 = future.result()
           store_shifts2[ff_idx] = x_shift2
    
    # interpolate zero shifts (which are where it's failed)
    np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_store_shifts2cc{num_folder:03d}.npy", store_shifts2)
    
    store_valid = store_shifts2!=0
    store_row = np.linspace(0,store_shifts2.shape[0],store_shifts2.shape[0]).reshape(-1,1)
    store_shifts2_valid = store_shifts2[store_valid]
    store_row_valid = store_row[store_valid]
    print(store_row_valid.shape, store_shifts2_valid.shape)
    if store_row_valid.shape[0] == 0 :
      print("skip")
      pass
    else :
      interp_func = interp1d(store_row_valid,  store_shifts2_valid, kind='slinear', bounds_error=False, fill_value=np.nan)
      store_shifts2 = interp_func(store_row)
      # #
      store_shifts2_x = np.linspace(0,store_shifts2.shape[0], store_shifts2.shape[0]).reshape(-1,1)
      valid = ~np.isnan(store_shifts2)
      # spline = UnivariateSpline(store_shifts2_x[valid], store_shifts2[valid], s=1e3, k=5,ext=0)  # s is the smoothing factor
      store_shifts2 = gaussian_filter1d(store_shifts2[valid], sigma=stack/0.5)
      # store_shifts2b = spline(store_shifts2_x)
      np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_store_shifts2cc{num_folder:03d}.npy", store_shifts2)
      
      # for ff in range(prof_store_stats_strain.shape[0]):  # basic loop... see below for parallelized
      def final_shift_correct(ff, total_size):    
          # TODO
          # update x0,y0 (and x1y1,x2y2) to incororate new shift, so we keep track of new fault position
          # print(f"processing line {ff} of {prof_store_stats_strain.shape[0]}")
          re_shift = np.full((3, prof_store_stats_strain.shape[2]), np.nan)
          # re_shift = np.zeros((3, prof_store_stats_strain.shape[2]))
       
          prof = ff   # prof = 2700 is where is collapses
          ysp = prof_store_stats_strain[prof,0,:]  # 0 = shear / 2 = dilatation
          yspd = prof_store_stats_strain[prof,2,:]  # 0 = shear / 2 = dilatation
          yspss = prof_store_stats_disp[prof,6,:]  # 0 = shear / 2 = dilatation
          xsp = prof_dist_pxls  # because the profiles has been shifted so the peak is a 1 (not zero)... at least, that's what I see in the syntethics!
          # w = smoothed_boxcar(plen, 9, 1) # (len, support_width, gaussian_sigma)
          #
          xsp2 = xsp
          ysp2 = ysp
          yspd2 = yspd
          yspss2 = yspss
          #
          valid = (np.isnan(ysp2) + np.isnan(yspd2) + np.isnan(yspss2)) < 1
          xsp2 = xsp2[valid]
          ysp2 = ysp2[valid]
          yspd2 = yspd2[valid]
          yspss2 = yspss2[valid]
          # w2 = w[valid]
          #
          # # re-shift profiles
          try:
              interp_func1 = interp1d(xsp2-store_shifts2[prof],  ysp2, kind='slinear', bounds_error=False, fill_value=np.nan)
              interp_func2 = interp1d(xsp2-store_shifts2[prof],  yspd2, kind='slinear', bounds_error=False, fill_value=np.nan)
              interp_func3 = interp1d(xsp2-store_shifts2[prof],  yspss2, kind='slinear', bounds_error=False, fill_value=np.nan)
              re_shift[0,:] = interp_func1(xsp)
              re_shift[1,:] = interp_func2(xsp)
              re_shift[2,:] = interp_func3(xsp)
              # switch nan to 0
              re_shift[0,:][np.isnan(re_shift[0,:])] = 0.
              re_shift[1,:][np.isnan(re_shift[1,:])] = 0.
              re_shift[2,:][np.isnan(re_shift[2,:])] = 0.
          except Exception as e:
              print(f"not enough good data to interpolate... skipping row {ff}")
          print(f"final shift correct... row: {ff} (of {total_size})")
          return ff, re_shift
      
      # parallelization
      with ProcessPoolExecutor(max_workers=cores) as executor1:
         futures1 = [executor1.submit(final_shift_correct, ff, prof_store_stats_disp.shape[0]) for ff in range(prof_store_stats_strain.shape[0])]
         for future in as_completed(futures1):
             ff_idx, re_shift = future.result()
             prof_store_stats_strain[ff_idx,0,:] = re_shift[0,:]
             prof_store_stats_strain[ff_idx,2,:] = re_shift[1,:]
             prof_store_stats_disp[ff_idx,6,:] = re_shift[2,:]
       
      
      # # save data ... aligned by shear
      print("saving re-shifed results")
      np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_straincc{num_folder:03d}.npy", prof_store_stats_strain)
      np.save(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_dispcc{num_folder:03d}.npy", prof_store_stats_disp)
    
      # np.save("profs_stack_1000_aligned_by_shear_data_store_shifts2cc.npy", store_shifts2) # infact, they aren't aligned at all
  
      
      prof_store_stats_strain = np.load(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_straincc{num_folder:03d}.npy")
      prof_store_stats_disp = np.load(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_dispcc{num_folder:03d}.npy")
      store_shifts = np.load(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_store_shiftsc{num_folder:03d}.npy")
      store_shifts_2 = np.load(f"{temp_folder}/profs_stack_1000_aligned_by_shear_data_store_shifts2cc{num_folder:03d}.npy")
      
      
      #%% 
      
      store = np.zeros((prof_store_stats_disp.shape[0], 42))
      store_shifted_profs = np.zeros((prof_store_stats_disp.shape[0], 3, prof_store_stats_disp.shape[2]))
      fault_pos = np.column_stack((x0,x1,x2,y0,y1,y2,angle))  # x0,y0 = on-fault, x1,y1=left, x2,y2=right
      fault_pos = fault_pos[stack:-stack,:]
      prof_dist_pxls = np.linspace(-plen, plen, num=(2*plen)+1)
      
      def compute_one(ff, skip, total_size):
         print(f"processing line {ff} of {total_size} ({int(ff/skip)} of {int(total_size/skip)} calculations)")
         # create stores
         result_store = np.zeros(42)
         result_shifted = np.zeros((3, prof_store_stats_disp.shape[2]))
      
         prof = ff   # prof = 2700 is where is collapses
         ysp = prof_store_stats_strain[prof,0,:]  # 0 = shear / 2 = dilatation
         yspd = prof_store_stats_strain[prof,2,:]  # 0 = shear / 2 = dilatation
         yspss = prof_store_stats_disp[prof,6,:]  # 0 = shear / 2 = dilatation
         xsp = prof_dist_pxls  # because the profiles has been shifted so the peak is a 1 (not zero)... at least, that's what I see in the syntethics!
         # plt.plot(xsp,ysp); plt.ylim(-0.001,0.0015); plt.plot([0,0],[-1,1]); #plt.xlim(-20,20)
      
         # define a weighting function for reducing noise around peak strain (so we can better locate it)
         # Set expected peak position and spread
         # expected_peak = plen
         # xx = np.arange((2*plen+1))
         # sigma = plen * 0.25  # adjust depending on how wide your peak region is
         # w = np.exp(-0.5 * ((xx - expected_peak) / sigma)**2)
         # w = np.ones_like(xsp)  # at this point, the heavy stacking should reduce the need to weight, thus I just use 1's
         # rounded boxcar weighting
         w = smoothed_boxcar(plen, 50, 1) # (len, support_width, gaussian_sigma)
         xsp2 = xsp
         ysp2 = ysp
         yspd2 = yspd
         yspss2 = yspss
      
         # # deconvolve! (VERY expensive)  (OR do I deconv on fine?... to deal with different smoothing/step_sampling window sizes better, AND get better resolution?)
         # n = len(ysp)
         # weights = np.ones(n - 1)
         # weights[:int(n*0.2)] = 1e-1
         # weights[-int(n*0.2):] = 1e-1
         # #
         # # careful...  weights, 1e-3 for mm, but 1e-5 for cc... needs tuning (controls how smooth the fits are)
         # ysp[np.isnan(ysp)] = 0.
         # res = minimize(objective, ysp, args=(ysp, weights, 1e-4), method='L-BFGS-B', tol=1e-10, options={'maxiter': 5000, 'maxfun': 10000}) # 5000 / 10000 (
         # ysp_dc = res.x
         # #
         # yspd[np.isnan(yspd)] = 0.
         # res = minimize(objective, yspd, args=(yspd, weights, 1e-4), method='L-BFGS-B', tol=1e-10, options={'maxiter': 5000, 'maxfun': 10000}) # 5000 / 10000 (
         # yspd_dc = res.x
         # #
         # yspss[yspss == 0] = np.nan  # first convert zeros to NaN
         # nans = np.isnan(yspss)
         # x = np.arange(len(yspss))
         # yspss[nans] = np.interp(x[nans], x[~nans], yspss[~nans])
         # res = minimize(objective, yspss, args=(yspss, weights, 5e-2), method='L-BFGS-B', tol=1e-10, options={'maxiter': 5000, 'maxfun': 10000}) # 5000 / 10000
         # yspss_dc = res.x
         # #
         # xsp2 = xsp
         # ysp2 = ysp_dc
         # yspd2 = yspd_dc
         # yspss2 = yspss_dc
         #
         # valid_all = np.sum((np.isnan(ysp2) + np.isnan(yspd2) + np.isnan(yspss2) + (ysp2==0) + (yspd2==0) + (yspss2==0)) <1)
         
         valid = (np.isnan(ysp2) + ysp2==0) < 1
         valid2 = (np.isnan(yspd2) + yspd2==0) < 1
         valid3 = (np.isnan(yspss2) + yspss2==0) < 1
         
         # only proceed if valid are not all nan
         if np.sum(valid) >= ysp2.shape[0]*0.5:
             for gg in [1]:
                 # proceed   
                 xsp2 = xsp[valid]
                 ysp2 = ysp[valid]
                 yspd2 = yspd[valid2]
                 yspss2 = yspss[valid3]
                 w2 = w[valid]
                 
                 if gg==1:
                     # detrend background strain
                     # ysp2_valid, _, _ = robust_subset(ysp2,0.7,min_points=15, fallback='median')  # function to find largest selection of points with minimum variance (a bit like ransac)
                     # ysp2_valid = ysp2<np.nanquantile(ysp,0.75)
                     #ysp2_valid = ((xsp2<=-32) | (xsp2>=20))
                     ysp2_valid = ((xsp2<=background_limit[0]) | (xsp2>=background_limit[1]))
                     ysp2_masked = ysp2[ysp2_valid]
                     xsp2_masked = xsp2[ysp2_valid]
                     xx = xsp2_masked.reshape(-1, 1)
                     yy = ysp2_masked.reshape(-1, 1)
                     poly_degree = 3
                 elif gg==2:
                     ysp2_valid = ((xsp2<=-x_minl) | (xsp2>=x_maxr))
                     ysp2_masked = ysp2[ysp2_valid]
                     xsp2_masked = xsp2[ysp2_valid]
                     xx = xsp2_masked.reshape(-1, 1)
                     yy = ysp2_masked.reshape(-1, 1)
                     poly_degree = 7
                 # Create and fit the RANSAC model... shear
                 try:
                     # poly_degree = 3  # not sure if high order is good?... it flattens better... lower order can leave negative residuals away from fault(?)
                     ransac_in = RANSACRegressor(
                         min_samples=0.6,
                         residual_threshold=0.2*np.std(ysp2_masked),  # This depends on your y scale; try values like 0.5, 1, 2
                         max_trials=100,
                         random_state=42
                         )
                     model = make_pipeline(PolynomialFeatures(poly_degree), ransac_in)
                     model.fit(xx, yy)
                     # Get the RANSAC regressor
                     ransac = model.named_steps['ransacregressor']
                     # Get inlier mask
                     inlier_mask = ransac.inlier_mask_
                     # Predict on all x
                     y_pred = model.predict(xx)
                     # Compute residuals only on inliers
                     residuals = yy[inlier_mask] - y_pred[inlier_mask]
                     # Standard deviation of residuals
                     std = np.std(residuals)
                     # compute r2
                     r2 = r2_score(yy[inlier_mask], y_pred[inlier_mask])
                     # fit curve
                     ysp_background = model.predict(xsp.reshape(-1,1))
                     ysp2_masked_background = model.predict(xsp2_masked.reshape(-1,1))
                     # detrend raw data
                     ysp_detrend = ysp.reshape(-1,1)-ysp_background
                     ysp2_masked_detrend = ysp2_masked.reshape(-1,1)-ysp2_masked_background
                     # recompute stdev on flattened strain profile, using the data used for detrending)
                     std2 = np.nanstd(ysp2_masked_detrend)
                     stdthr = stdthr
                     ysp_threshold = np.zeros_like(ysp) + (stdthr*std2)  # 1 std = 68% / 1.28 = 80% / 1.645 = 90% / 2 = 95% / 2.3 = 98% / 2.576 = 99% / 3 = 99.7%
                     threshold = stdthr*std2
                     threshold2 = threshold
                     if gg==1:
                         # find current xmin and xmax
                         # left
                         crossing_idx_left = np.where(xsp<=0)
                         xsp_left = (-xsp[crossing_idx_left])[::-1]
                         ysp_detrend_left = (ysp_detrend[crossing_idx_left])[::-1]
                         # Find indices where y is below the threshold
                         indices_below = np.where(ysp_detrend_left < threshold)[0]
                         x_minl = -(xsp_left[indices_below[0]])#-0.5  # Smallest x where y is below threshold
                         # x_min_idx = np.where(xsp==x_min)
                         # right
                         crossing_idx_right = np.where(xsp>=0)
                         xsp_right = (xsp[crossing_idx_right])[::1]
                         ysp_detrend_right = (ysp_detrend[crossing_idx_right])[::1]
                         # Find indices where y is below the threshold
                         indices_below = np.where(ysp_detrend_right < threshold)[0]
                         x_maxr = xsp_right[indices_below[0]]
                         # x_max_idx = np.where(xsp==x_max0)
                         ysp = np.squeeze(ysp_detrend)  
                     elif gg==2:
                         ysp = np.squeeze(ysp_detrend)   
                         
                 except Exception as e:
                     threshold = 0
                     threshold2 = threshold
                     r2 = np.nan
                     print(f"couldn't detrend strain, row {ff}")
                 
             # now re-do, using a detrending that is more tailored to the data ()
             
             # # # deconvolve
             # # n = len(ysp)
             # # weights = np.ones(n - 1)*1
             # # weights[:int(n*0.2)] = 1e-1
             # # weights[-int(n*0.2):] = 1e-1
             
             # # weights = np.ones(n - 1)
             # # weights[:] = 1.0  # default
             # # # Allow *almost no penalty* near the spike
             # # weights[spike_center - spike_width : spike_center + spike_width] = 1e-10
             # weights=(np.nanmax(ysp)*(1-smoothed_boxcar(len(ysp)/2-1.,3,1)))+1e-14
             # #
             # # careful...  weights, 1e-3 for mm, but 1e-5 for cc... needs tuning (controls how smooth the fits are)
             # ysp[np.isnan(ysp)] = 0.
             # res = minimize(objective, ysp, args=(ysp, weights, 5e-2), method='L-BFGS-B', tol=1e-12, options={'maxiter': 100, 'maxfun': 15000}) # 5000 / 10000 (
             # ysp_dc = res.x
             
             #
             ysp2 = ysp[valid]
             # ysp2 = ysp_dc[valid]
             #
             # Define a fine range of x values for evaluation
             xno = int(xsp.shape[0]*1e3)
             x_fine = np.linspace(xsp.min(), xsp.max(), xno)
             try:
                 spline = UnivariateSpline(xsp2, ysp2*(w[valid]), s=1e-9)  # s is the smoothing factor 1.5e7
                 y_fine = spline(x_fine)
             except Exception as e:
                 y_fine = np.zeros_like(x_fine)
             try:
                 spline2 = UnivariateSpline(xsp2, yspd2*(w[valid2]), s=1e-9)  # s is the smoothing factor 1.5e7
                 y2_fine = spline2(x_fine)
             except Exception as e:
                 y2_fine = np.zeros_like(x_fine)
             
              # spline3 = UnivariateSpline(xsp2, yspss2, s=1e-8)  # s is the smoothing factor
             # y3_fine = spline3(x_fine)
             # compute remaining shift
             try:
                 x_shift = (np.nanargmax(y_fine)/xno)*xsp.shape[0] - (xsp.shape[0]/2) # minus 0.5, because the peak is interpolated between 0 and 1
                 print(f"profile shift: {x_shift}, row: {prof}")
             except Exception as e:
                 x_shift = 0
                 print(f"profile shift: {x_shift} (all nan!), row: {prof}")
             y_fine[np.isnan(y_fine)] == 0.
             y2_fine[np.isnan(y2_fine)] == 0.
             # # re-shift profiles
             # interp_func1 = interp1d(xsp2-x_shift,  ysp2, kind='slinear', bounds_error=False, fill_value=np.nan)
             # interp_func2 = interp1d(xsp2-x_shift,  yspd2, kind='slinear', bounds_error=False, fill_value=np.nan)
             # interp_func3 = interp1d(xsp2-x_shift,  yspss2, kind='slinear', bounds_error=False, fill_value=np.nan)
             # ysp_dc = interp_func1(xsp)
             # yspd_dc = interp_func2(xsp)
             # yspss_dc = interp_func3(xsp)
             # # switch nan to 0
             # ysp_dc[np.isnan(ysp_dc)] = 0.
             # yspd_dc[np.isnan(yspd_dc)] = 0.
             # yspss_dc[np.isnan(yspss_dc)] = 0.
             #
             result_shifted[0,:] = ysp # ysp_dc # shear
             result_shifted[1,:] = yspd #yspd_dc # dilat
             result_shifted[2,:] = yspss #prof_store_stats_strain[prof,0,:] #yspss2 #yspss_dc # flt parallel
             #
             
          
             # threshold = 0.0035  # shear strain threshold
             # threshold2 = 0.001  # dilatational strain threshold
             # adjust x to account for shift
             # x_fine = x_fine-(0.5*xno)
             # left side
             try:
                 crossing_idx_left = np.where(x_fine<=0)
                 x_fine_left = (-x_fine[crossing_idx_left])[::-1]
                 y_fine_left = (y_fine[crossing_idx_left])[::-1]
                 # Find indices where y is below the threshold
                 indices_below = np.where(y_fine_left < threshold)[0]
                 x_min = -(x_fine_left[indices_below[0]])#-0.5  # Smallest x where y is below threshold
             except IndexError as e:
                 x_min = np.nan
          
             # right side
             try:
                 crossing_idx_right = np.where(x_fine>=0)
                 x_fine_right = (x_fine[crossing_idx_right])[::1]
                 y_fine_right = (y_fine[crossing_idx_right])[::1]
                 # Find indices where y is below the threshold
                 indices_below = np.where(y_fine_right < threshold)[0]
                 x_max = x_fine_right[indices_below[0]]#-0.5  # Smallest x where y is below threshold
                 fzw = x_max-x_min  # x_max is positive, x_min is negative, thus x_max-x_min gives full difference
             except IndexError as e:
                 x_max = np.nan
                 fzw = np.nan
          
             # dilatation
             # left side
             try:
                 crossing_idx_left2 = np.where(x_fine<=0)
                 x_fine_left2 = (-x_fine[crossing_idx_left2])[::-1]
                 y_fine_left2 = (y2_fine[crossing_idx_left2])[::-1]
                 # Find indices where y is below the threshold
                 indices_below2 = np.where(y_fine_left2 < threshold2)[0]
                 x_min2 = -(x_fine_left2[indices_below2[0]])#-0.5  # Smallest x where y is below threshold
             except IndexError as e:
                 x_min2 = np.nan
          
             # right side
             try:
                 crossing_idx_right2 = np.where(x_fine>=0)
                 x_fine_right2 = (x_fine[crossing_idx_right2])[::1]
                 y_fine_right2 = (y2_fine[crossing_idx_right2])[::1]
                 # Find indices where y is below the threshold
                 indices_below2 = np.where(y_fine_right2 < threshold2)[0]
                 x_max2 = x_fine_right2[indices_below2[0]]#-0.5  # Smallest x where y is below threshold
                 fzw2 = x_max2-x_min2
             except IndexError as e:
                 x_max2 = np.nan
                 fzw2 = np.nan
             #
             ## calculate offset by near-field regression
             fperp_dist = 15 # in pixels, thus 15 * 10 = 150m vs 15 * 40 = 600  #/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ # TO MODIFY
             fperp_dist_exp  = 3                                                #/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ # TO MODIFY
             
             fperp_dist = near_field_limit[1]
             fperp_dist_exp = near_field_limit[0]
          #   residual_threshold = 0.1
             # right
             limits_idx_right = np.where( (xsp2 >= x_max+fperp_dist_exp) & (xsp2 <= x_max+fperp_dist) )
             x_right = xsp2[limits_idx_right]
             y_right = yspss2[limits_idx_right]
             residual_threshold = np.std(y_right)*3
             # Fit robust linear regression
             ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold) # sklearn expects x_right to be a 2D array
             try:
                 ransac.fit(x_right.reshape(-1, 1), y_right)
                 # Predict values
                 xr_pred = np.linspace(0, x_right.max(), xsp.shape[0]).reshape(-1, 1)  # Create a smooth range of x values
                 yr_pred = ransac.predict(xr_pred)  # Predict y values
                 # stdev
                 yr_pred_raw = ransac.predict(x_right.reshape(-1, 1))  # Predict y values
                 residuals_r = y_right - yr_pred_raw # residuals
                 stdev_r = np.std(residuals_r)
             except ValueError as e:
                 yr_pred = [0,0]
                 stdev_r = 0
                 x_right = [0]
                 y_right = [0]
          
             # left
             limits_idx_left = np.where( (xsp2 <= x_min-fperp_dist_exp) & (xsp2 >= x_min-fperp_dist) )
             x_left = xsp2[limits_idx_left]
             y_left = yspss2[limits_idx_left]
             residual_threshold = np.std(y_left)*3
             # Fit robust linear regression
             ransac2 = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold) # sklearn expects x_right to be a 2D array
             try:
                 ransac2.fit(x_left.reshape(-1, 1), y_left)
                 # Predict values
                 xl_pred = np.linspace(x_left.min(), 0, xsp.shape[0]).reshape(-1, 1)  # Create a smooth range of x values
                 yl_pred = ransac2.predict(xl_pred)  # Predict y values
                 # stdev
                 yl_pred_raw = ransac2.predict(x_left.reshape(-1, 1))  # Predict y values
                 residuals_l = y_left - yl_pred_raw # residuals
                 stdev_l = np.std(residuals_l)
             except ValueError as e:
                 yl_pred = [0,0]
                 stdev_l = 0
                 x_left = [0]
                 y_left = [0]
          
             # offset
             offset =  yr_pred[0] - yl_pred[-1]
             # Calculate combined standard deviation (assuming independent errors)
             combined_std_dev = np.sqrt(stdev_r**2 + stdev_l**2)
             #
          
             ###
             ## calculate offset by FAR-field regression
             #
             fperp_dist = 25 #plen                    #/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ # TO MODIFY
             fperp_dist_exp  = 10 #int(plen/2)        #/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ # TO MODIFY
             fperp_dist = far_field_limit[1]
             fperp_dist_exp = far_field_limit[0]
          #   residual_threshold = 0.1
          
             # right
             limits_idx_right = np.where( (xsp2 >= x_max+fperp_dist_exp) & (xsp2 <= x_max+fperp_dist) )
             x_right2 = xsp2[limits_idx_right]
             y_right2 = yspss2[limits_idx_right]
             residual_threshold = np.std(y_right2)*3
             # Fit robust linear regression
             ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold) # sklearn expects x_right to be a 2D array
             try:
                 ransac.fit(x_right2.reshape(-1, 1), y_right2)
                 # Predict values
                 xr_pred = np.linspace(0, x_right2.max(), xsp.shape[0]).reshape(-1, 1)  # Create a smooth range of x values
                 yr_pred = ransac.predict(xr_pred)  # Predict y values
                 # stdev
                 yr_pred_raw = ransac.predict(x_right2.reshape(-1, 1))  # Predict y values
                 residuals_r = y_right2 - yr_pred_raw # residuals
                 stdev_r = np.std(residuals_r)
             except ValueError as e:
                 x_right2 = [0]
                 y_right2 = [0]
                 yr_pred = [0,0]
                 stdev_r = 0
          
             # left
             limits_idx_left = np.where( (xsp2 <= x_min-fperp_dist_exp) & (xsp2 >= x_min-fperp_dist) )
             x_left2 = xsp2[limits_idx_left]
             y_left2 = yspss2[limits_idx_left]
             residual_threshold = np.std(y_left2)*3
             # Fit robust linear regression
             ransac2 = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold) # sklearn expects x_right to be a 2D array
             try:
                 ransac2.fit(x_left2.reshape(-1, 1), y_left2)
                 # Predict values
                 xl_pred = np.linspace(x_left2.min(), 0, xsp.shape[0]).reshape(-1, 1)  # Create a smooth range of x values
                 yl_pred = ransac2.predict(xl_pred)  # Predict y values
                 # stdev
                 yl_pred_raw = ransac2.predict(x_left2.reshape(-1, 1))  # Predict y values
                 residuals_l = y_left2 - yl_pred_raw # residuals
                 stdev_l = np.std(residuals_l)
             except ValueError as e:
                 x_left2 = [0]
                 y_left2 = [0]
                 yl_pred = [0,0]
                 stdev_l = 0
          
             # offset
             offset2 = yr_pred[0] - yl_pred[-1]
             # Calculate combined standard deviation (assuming independent errors)
             combined_std_dev2 = np.sqrt(stdev_r**2 + stdev_l**2)
          
             ###
             ## calculate offset by NEAR-field regression : dilatation
             # yspd = prof_store_stats_disp[prof,6,:] # fault-parallel displacements
             #
             fperp_dist = 15
             fperp_dist_exp  = 3
             fperp_dist = near_field_limit[1]                         #/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ # TO MODIFY
             fperp_dist_exp = near_field_limit[0]                     #/!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ /!\ # TO MODIFY
          #   residual_threshold = 0.1
          
             # right
             limits_idx_right = np.where( (xsp2 >= x_max2+fperp_dist_exp) & (xsp2 <= x_max2+fperp_dist) )
             x_right3 = xsp[limits_idx_right]
             y_right3 = yspss2[limits_idx_right]
             residual_threshold = np.std(y_right3)*3
          
             # Fit robust linear regression
             ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold) # sklearn expects x_right to be a 2D array
             try:
                 ransac.fit(x_right3.reshape(-1, 1), y_right3)
                 # Predict values
                 xr_pred = np.linspace(0, x_right3.max(), xsp.shape[0]).reshape(-1, 1)  # Create a smooth range of x values
                 yr_pred = ransac.predict(xr_pred)  # Predict y values
                 # stdev
                 yr_pred_raw = ransac.predict(x_right3.reshape(-1, 1))  # Predict y values
                 residuals_r = y_right3 - yr_pred_raw # residuals
                 stdev_r = np.std(residuals_r)
             except ValueError as e:
                 x_right3 = [0]
                 y_right3 = [0]
                 yr_pred = [0,0]
                 stdev_r = 0
          
             # left
             limits_idx_left = np.where( (xsp2 <= x_min2-fperp_dist_exp) & (xsp2 >= x_min2-fperp_dist) )
             x_left3 = xsp2[limits_idx_left]
             y_left3 = yspss2[limits_idx_left]
             residual_threshold = np.std(y_left3)*3
             # Fit robust linear regression
             ransac2 = RANSACRegressor(estimator=LinearRegression(), residual_threshold=residual_threshold) # sklearn expects x_right to be a 2D array
             try:
                 ransac2.fit(x_left3.reshape(-1, 1), y_left3)
                 # Predict values
                 xl_pred = np.linspace(x_left3.min(), 0, xsp.shape[0]).reshape(-1, 1)  # Create a smooth range of x values
                 yl_pred = ransac2.predict(xl_pred)  # Predict y values
                 # stdev
                 yl_pred_raw = ransac2.predict(x_left3.reshape(-1, 1))  # Predict y values
                 residuals_l = y_left3 - yl_pred_raw # residuals
                 stdev_l = np.std(residuals_l)
             except ValueError as e:
                 x_left3 = [0]
                 y_left3 = [0]
                 yl_pred = [0,0]
                 stdev_l = 0
          
             # offset
             offset3 = yr_pred[0] - yl_pred[-1]
             # Calculate combined standard deviation (assuming independent errors)
             combined_std_dev3 = np.sqrt(stdev_r**2 + stdev_l**2)
             #
             # print("offset:", f"{offset:.2f}", "+/-", f"{combined_std_dev:.3f}", "meters")
         else:
             offset = np.nan()
             combined_std_dev = np.nan()
             x_left = np.nan()
             x_left = np.nan()
             x_right = np.nan()
             x_right = np.nan()
             offset2 = np.nan()
             combined_std_dev2 = np.nan()
             x_left2 = np.nan()
             x_left2 = np.nan()
             x_right2 = np.nan()
             x_right2 = np.nan()
             offset3 = np.nan()
             combined_std_dev3 = np.nan()
             x_left3 = np.nan()
             x_left3 = np.nan()
             x_right3 = np.nan()
             x_right3 = np.nan()
             fzw = np.nan()
             x_min = np.nan()
             x_max = np.nan()
             threshold = np.nan()
             fzw2 = np.nan()
             x_min2 = np.nan()
             x_max2 = np.nan()
             threshold2 = np.nan()
             ysp = np.nan()
             yspd = np.nan()
             x_shift = np.nan()
             threshold = np.nan()
      
         # store results
         A_fit, mu_fit, sigma_fit, alpha_fit, b_fit = 0, 0, 0, 0, 0
         r_squared = 0
      
         result_store = [x0[int(ff+stack)], y0[int(ff+stack)], angle[int(ff+stack)], \
                         (transform*(x0[int(ff+stack)],y0[int(ff+stack)]))[0], \
                             (transform*(x0[int(ff+stack)],y0[int(ff+stack)]))[1], \
                                 offset, combined_std_dev, x_left[0], x_left[-1], x_right[0], x_right[-1], \
                                     offset2, combined_std_dev2, x_left2[0], x_left2[-1], x_right2[0], x_right2[-1], \
                                         offset3, combined_std_dev3, x_left3[0], x_left3[-1], x_right3[0], x_right3[-1], \
                                             fzw*xres, x_min*xres, x_max*xres, threshold, \
                                                 fzw2*xres, x_min2*xres, x_max2*xres, threshold2, \
                                                     np.nanmax(ysp), np.nanmax(yspd), x_shift, r2, \
                                                         A_fit, mu_fit, sigma_fit, alpha_fit, b_fit, r_squared, (sigma_fit*2.05)*2*xres]
         return ff, result_store, result_shifted
      
      
      # parallelization
      skip=1
      with ProcessPoolExecutor(max_workers=cores) as executor:
         futures = [executor.submit(compute_one, ff, skip, prof_store_stats_disp.shape[0]) for ff in range(0,prof_store_stats_disp.shape[0],skip)]
         for future in as_completed(futures):
             ff_idx, result_store, result_shifted = future.result()
             store[ff_idx] = result_store
             store_shifted_profs[ff_idx] = result_shifted
      
      
      
      # save data
      np.save(f"{temp_folder}/store_data_stack_1000_alignment_peak_shear__shear_0.0035_dilatation_0.001b{num_folder:03d}.npy", store)
      np.save(f"{temp_folder}/store_shifted_profs_data_stack_1000_alignment_peak_shear__shear_0.0035_dilatation_0.001b{num_folder:03d}.npy", store_shifted_profs)
      
      #store = np.load(f"{temp_folder}/store_data_stack_1000_alignment_peak_shear__shear_0.0035_dilatation_0.001b{num_folder:03d}.npy")
      #store_shifted_profs = np.load(f"{temp_folder}/store_shifted_profs_data_stack_1000_alignment_peak_shear__shear_0.0035_dilatation_0.001b{num_folder:03d}.npy")
      # # thin
      #store = store[::skip,:]
      #store_shifted_profs = store_shifted_profs[::skip,:]
      
      create_shp(transform, fault_pos[:,0:2], name="central_profile_points", 
                    shp_type="point", nb_profile=num_folder, nb_seg=num_folder,
                    folder=temp_folder)   
      create_shp(transform, fault_pos[:,2:6], name="profiles", shp_type="line",
                    nb_profile=num_folder, nb_seg=num_folder, 
                    folder=temp_folder)   
      
    num_folder = num_folder + 1
t1 = time.time()
print("durée :", t1-t0)