from pywordseg import Wordseg


class ChineseSegmenter:
    def segment_data(self, data):
        seg = Wordseg(
            batch_size=64,
            device="cpu",
            embedding="elmo",
            elmo_use_cuda=False,
            mode="TW",
        )

        segmented = seg.cut([data["text"]])

        processed = ""

        for i in range(0, len(segmented)):
            for j in range(0, len(segmented[i])):
                processed = processed + segmented[i][j] + " "
        data["text"] = processed

        return data
