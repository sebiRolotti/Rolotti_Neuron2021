from __future__ import print_function
import numpy as np
from ExperimentAbstractClasses import TargetingMixin
from Experiments import InductionExperiment

from scipy.ndimage import center_of_mass
from sima.ROI import mask2poly
from shapely.geometry import Point

from lab.misc.misc import unique_rows


class SingleCellExperiment(InductionExperiment.InductionExperiment, TargetingMixin.TargetingMixin):
    """

    Single Cell Experiment Implementation.

    """

    def __init__(self, trial_id, mark_points_path):
        super(SingleCellExperiment, self).__init__(trial_id)

        self._mark_points_path = mark_points_path
        self._initialize_xml_formatter()
        self._time_average = self.imaging_dataset().time_averages[0, ..., -1]

    # ****************************************************************
    # * Targeting Mixin  overwritten methods
    # *****************************************************************

    @staticmethod
    def _euclidean_distance(p1, p2):
        return np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(p1, p2)]))

    def _closest_cell(self, centroids, point):
        return np.argmin([self._euclidean_distance(c, point) for c in centroids])

    @staticmethod
    def _com(roi):
        return center_of_mass(np.array(roi.mask[0].todense()))

    def find_targeted_cells(self, label='s2p'):
        # This should return all cells ever with zero distance of stimulation

        rois = self.rois(label=label)
        centroids = [self._com(roi) for roi in rois]

        idx = [self._closest_cell(centroids, point) for point in self.stim_locations]
        return idx

    def _all_stim_locations(self):

        # This is the location of the stimulation relative to the time-average
        stim_loc = self.stim_locations[0, :]

        # Use the sequence information to figure out exactly where
        # stimulations actually occurred
        seq = self.imaging_dataset().sequences[0]

        # Now find all the displacements during stim frames
        indices = seq._indices
        ymin = indices[2].start
        xmin = indices[3].start
        offset = np.array([ymin, xmin])

        stim_duration = int(self._get_stim_duration() / self.frame_period())
        stim_starts = self._get_stim_frames()

        disps = seq.displacements
        all_disps = []
        for stim_start in stim_starts:
            all_disps.extend(disps[stim_start:stim_start + stim_duration, 0, :])
        all_disps = np.vstack(all_disps) - offset

        # We now have the true location, in pixels, of the stim center during all frames
        # Note we subtract the displacements because the stim location is "moving"
        # oppositely relative to frame movement
        true_locs = stim_loc - all_disps

        return true_locs

    def _roi_dists(self, roi_filter=None):
        # Return minimum distance of all rois to each location

        locs = self._all_stim_locations()
        locs = unique_rows(locs)

        spiral_height = self.spiral_sizes[0] * self.frame_shape()[2]
        all_stims = []
        for loc in locs:
            spiral_center = Point(loc)
            spiral_footprint = spiral_center.buffer(spiral_height / 2.)
            all_stims.append(spiral_footprint)

        # Now Calculate distances for all ROIs
        rois = self.rois(roi_filter=roi_filter)
        dists = []
        for roi in rois:
            roi_poly = mask2poly(self.binarize_mask(roi).T)
            dist = np.min([roi_poly.distance(spiral) for spiral in all_stims])
            dists.append(dist)

        return dists

    def find_targeted_cells_full(self, roi_filter=None):

        dists = self._roi_dists(roi_filter=roi_filter)

        return [i for i, d in enumerate(dists) if d == 0]

    @staticmethod
    def binarize_mask(roi, thres=0.2):
        mask = np.sum(roi.__array__(), axis=0)
        return mask > (np.max(mask) * thres)


    # ****************************************************************
    # * Stimulation Experiment overwritten methods
    # *****************************************************************

    def _parse_context(self):
        contexts = self.behaviorData()['__trial_info']['contexts']
        contexts = {k: contexts[k] for k in contexts
                    if k.startswith('led')}
        return contexts

    def _get_stim_frames(self):
        stim_frames = self._get_stim_times()[:, 0] / self.frame_period()
        return stim_frames.astype(int)

    def _get_stim_times(self):
        behaviour_data = self.behaviorData()
        start_times = behaviour_data['led_context_pin'][:, 0][1:]
        end_times = start_times + self._get_stim_duration()
        stacked = np.vstack((start_times, end_times)).T
        return stacked

    def _get_stim_duration(self):
        return self._xml_formatter.stim_duration

    # ****************************************************************
    # * Induction Experiment overwritten methods
    # *****************************************************************

    def _get_stim_positions(self, units='mm'):
        contexts = self._parse_context()
        positions = [contexts[k]['locations'][0] - contexts[k]['radius']
                     for k in contexts]

        if units == 'mm':
            return positions
        elif units == 'normalized':
            return [(x * 100.) / self.track_length for x in positions]
        else:
            print('Units not recognized')
            return None

    def _get_laps(self):
        pass

    def _get_session(self, session, markpoints=None):

        assoc_dict = self.get('assoc_expts', {})
        c, s = session.split('_')

        try:
            if markpoints:
                mp = markpoints
            elif c == 'control':
                # TODO add this during stim cell finder script
                mp = self.control_markpoints_path
            else:
                mp = self.cno_markpoints_path
        except KeyError:
            if (c == 'control') and ('control' in self.session):
                self.control_markpoints_path = self._mark_points_path
                self.save(store=True)
                mp = self.control_markpoints_path
            elif (c == 'cno') and ('cno' in self.session):
                self.cno_markpoints_path = self._mark_points_path
                self.save(store=True)
                mp = self.cno_markpoints_path
            else:
                print('No {} markpoints for expt {}'.format(c, self.trial_id))

        try:
            return self.__class__(assoc_dict[c][s], mp)
        except KeyError:
            print('{} not in association dictionary {}'.format(
                session, assoc_dict))
            return None

    # ****************************************************************
    # * Class Specific Methods
    # ****************************************************************

    @property
    def target_idx(self):
        rois = self.rois()
        idx = [i for i, r in enumerate(rois) if 'targeted' in r.tags]
        return idx[0]

    @property
    def stim_idx(self):
        rois = self.rois()
        return [i for i, r in enumerate(rois) if 'stimmed' in r.tags]
