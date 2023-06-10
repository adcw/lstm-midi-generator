from keras.layers import Layer, Conv1D, LSTM, Concatenate, Dense


class DeepLSTM(Layer):
    """
    Deep LSTM layer, using multiple combined Conv1D layers.
    """

    def __init__(self, filters: list[int] | None = None, kernels: list[int] | None = None, **kwargs):
        """
        Initialize DeepLSTM
        :param filters: list of filter sizes
        :param kernels: list of kernel sizes
        :param kwargs: kwargs of keras Layer
        """
        super(DeepLSTM, self).__init__(**kwargs)

        if kernels is None:
            kernels = [11, 19]

        if filters is None:
            filters = [256, 256]

        if len(kernels) != len(filters):
            raise ValueError(
                f'The filter and kernel lists must have same length, '
                f'received kernels = {len(kernels)}, filters = {len(filters)}'
            )

        self.filters = filters
        self.kernels = kernels
        self.conv_layers = []
        self.lstm_layers = []

        for i in range(len(filters)):
            self.conv_layers.append(Conv1D(filters=filters[i], kernel_size=kernels[i]))
            self.lstm_layers.append(LSTM(128, dropout=0.2, return_sequences=True))
            self.lstm_layers.append(LSTM(64, dropout=0.2))

        self.lstm_out1 = LSTM(256, dropout=0.2, return_sequences=True)
        self.lstm_out2 = LSTM(256, dropout=0.2, return_sequences=True)
        self.lstm_out3 = LSTM(128, dropout=0.2)

        self.concat = Concatenate()
        self.dense = Dense(128)
        self.dense2 = Dense(64)

    def call(self, inputs, *args, **kwargs):
        conv_outputs = []
        for i in range(len(self.filters)):
            conv_output = self.conv_layers[i](inputs)
            lstm_output = self.lstm_layers[i * 2](conv_output)
            lstm_output = self.lstm_layers[i * 2 + 1](lstm_output)
            conv_outputs.append(lstm_output)

        lstm_out1_output = self.lstm_out1(inputs)
        lstm_out2_output = self.lstm_out2(lstm_out1_output)
        lstm_out3_output = self.lstm_out3(lstm_out2_output)

        concat_output = self.concat([lstm_out3_output] + conv_outputs)

        dense_output = self.dense(concat_output)
        dense2_output = self.dense2(dense_output)

        return dense2_output
