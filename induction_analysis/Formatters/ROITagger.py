import cPickle as pkl


class ROITagger:

    def __init__(self, experiment, label='s2p'):
        self._experiment = experiment
        self._label = label

        self._rois = self._load_rois()

        self._tagged = {}
        self._untagged = {}

    def _load_rois(self):

        with open(self._experiment.signalsFilePath(), 'rb') as fp:

            rois = pkl.load(fp)[self._label]['rois']

        return rois

    def _save_rois(self):

        with open(self._experiment.signalsFilePath(), 'rb') as fp:

            sigs = pkl.load(fp)

        sigs[self._label]['rois'] = self._rois

        with open(self._experiment.signalsFilePath(), 'wb') as fw:
            pkl.dump(sigs, fw)

    def tag(self, idx, tags):

        if not isinstance(tags, list):
            tags = [tags] * len(idx)

        for i, tag in zip(idx, tags):

            if tag in self._rois[i]['tags']:
                continue

            self._rois[i]['tags'].add(tag)

            roi_label = self._rois[i]['label']
            try:
                self._tagged[roi_label].append(tag)
            except KeyError:
                self._tagged[roi_label] = [tag]

    def delete(self, tags, idx=None):
        if not idx:
            idx = range(len(self._rois))

        tags = set(tags)

        for i in idx:

            oldtags = self._rois[i]['tags']
            newtags = oldtags - tags

            if len(newtags) != len(oldtags):
                self._rois[i]['tags'] = newtags
                roi_label = self._rois[i]['label']
                removed = oldtags.intersection(tags)

                try:
                    self._untagged[roi_label].union(removed)
                except KeyError:
                    self._untagged[roi_label] = removed

    def save(self):

        self._save_rois()

    def print_changes(self):

        if not len(self._tagged) and not len(self._untagged):
            print 'No changes to ROI tags so far'
        else:
            for k, v in self._tagged.iteritems():
                print 'ROI {} has new tag(s): {}'.format(k, ', '.join(v))
            for k, v in self._untagged.iteritems():
                print 'ROI {} lost tag(s): {}'.format(k, ', '.join(v))
