class CloudCastOptions:
    def __init__(self, **kwargs):
        try:
            label = kwargs["label"]
            if label is not None:
                self.from_label(label)
            else:
                raise KeyError("")
        except KeyError as e:
            self.model = kwargs.get("model", "unet")
            self.n_channels = kwargs.get("n_channels", 1)
            self.loss_function = kwargs.get("loss_function", "MeanSquaredError")
            self.preprocess = kwargs.get("preprocess", "img_size=128x128")
            self.onehot_encoding = kwargs.get("onehot_encoding", False)
            self.leadtime_conditioning = kwargs.get("leadtime_conditioning", 0)

    def __str__(self):
        return self.get_label()

    def from_label(self, label):
        elems = label.split("-")

        self.model = elems[0]
        self.loss_function = elems[1]
        self.n_channels = int(elems[2].split("=")[1])
        
        # Bỏ qua các tính năng không cần thiết
        idx = 3
        while idx < len(elems):
            if elems[idx].startswith("lc="):
                self.leadtime_conditioning = int(elems[idx].split("=")[1])
                idx += 1
            elif elems[idx].startswith("oh="):
                self.onehot_encoding = eval(elems[idx].split("=")[1])
                idx += 1
            elif elems[idx].startswith("img_size="):
                self.preprocess = elems[idx]
                idx += 1
            else:
                idx += 1

    def get_label(self):
        return "{}-{}-hist={}-lc={}-oh={}-{}".format(
            self.model,
            self.loss_function,
            self.n_channels,
            self.leadtime_conditioning,
            self.onehot_encoding,
            self.preprocess,
        )
