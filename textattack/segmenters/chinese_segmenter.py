from pywordseg import Wordseg


class ChineseSegmenter:
    def segment_by_word(self, data):
        """
        segments data by chinese words
        https://pypi.org/project/pywordseg/
        """
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

    def segment_by_character(self, data):
        """segments data by chinese characters"""

        segmented_data = []
        for sentence in data["text"]:
            segmented_sentence = []
            for character in sentence:
                segmented_sentence.append(character)
            segmented_data.append(segmented_sentence)

        processed = ""

        for i in range(0, len(segmented_data)):
            for j in range(0, len(segmented_data[i])):
                processed = processed + segmented_data[i][j] + " "

        data["text"] = processed
        return data