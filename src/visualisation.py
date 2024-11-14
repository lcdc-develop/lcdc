import matplotlib.pyplot as plt
import numpy as np

from utils import Track, StrVars


def plot_track(track, mag=True, phase=False, dist=False, fourier=False):
    n_plots = sum([mag, phase, dist, fourier])
    height_ratios = [r for r, b in zip([6,2,4,4], [mag, fourier, phase, dist]) if b]

    fig, axs = plt.subplots(n_plots,1, gridspec_kw={"height_ratios": height_ratios}, figsize=(10, 2*n_plots))
    # fig, axs = plt.subplots(n_plots,1, gridspec_kw={"height_ratios": [4,1,4,4]}, figsize=(10, 2*n_plots))

    idx = 0
    axs[-1].set_xlabel("Time [sec]")
    if mag:
        axs[idx].scatter(track.data[:, 0], track.data[:,1], s=1, c='blue')
        axs[idx].invert_yaxis()
        axs[idx].set_title(f"Track {track.id}, NORAD: {track.norad_id}")
        axs[idx].set_ylabel("Magnitude")
        idx += 1

    if fourier:
        if StrVars.FOURIER_COEFS not in track.stats:
            track.stats["fourier_coefs"], _ = track.get_fourier_coeficients(8)

        order = len(track.stats["fourier_coefs"]) // 2
        t = np.linspace(0, track.data[-1,0], len(track.data))
        data = np.array([track._fourier(order, track.period)(x, *track.stats["fourier_coefs"]) for x in t])
        axs[0].plot(t, data, c='red', label="Fourier series")

        # plot residuals
        axs[idx].scatter(track.data[:, 0], track.data[:,1] - data, s=1, c='red')
        axs[idx].invert_yaxis()
        axs[idx].set_ylabel("Δ Magnitude")
        idx += 1
    
     
    if phase:
        axs[idx].scatter(track.data[:, 0], track.data[:,2], s=1, c='green')
        axs[idx].invert_yaxis()
        axs[idx].set_ylabel("Phase [Deg]")
        idx += 1
    
    if dist:
        axs[idx].scatter(track.data[:, 0], track.data[:,3], s=1, c='orange')
        axs[idx].invert_yaxis()
        axs[idx].set_ylabel("Distance [km]")
        idx += 1
    plt.tight_layout()
    
    
    return fig, axs


        

        
    
if __name__ == "__main__":

    # track = Track(id=19271946,
    #               norad_id=5,
    #               timestamp=0,mjd=0,period=0)
    ID = 19290843
    NORAD_ID = 29657
    track = Track(id=ID,
                  norad_id=NORAD_ID,
                  timestamp=0,mjd=0,period=135.402858360)
    
    track.load_data_from_file("/home/k/kyselica12/work/mmt/MMT/data")

    fig, axs = plot_track(track, mag=True, phase=True, dist=True, fourier=True)

    plt.savefig("track.png")
    print("Saved to track.png")
    
    #TODO:
    # zavislost magnitudy a fazy