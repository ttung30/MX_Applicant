import tensorflow as tf
class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel, strides, padding):
        '''
        Khởi tạo Convolution Block với các tham số đầu vào

        Parameters
        ----------
        filters: int
            số lượng filter
        kernel: int
            kích thước kernel
        strides: int
            stride của convolution layer
        padding: str
            Loại padding của convolution layer

        '''
        super(ConvBlock, self).__init__()
        ## TODO 2
        ### START CODE HERE
        self.cnn = tf.keras.layers.Conv2D(filters, kernel, strides=strides,  padding=padding, activation = 'relu')
        ### END CODE HERE
    def call(self, inputs):
        '''
        Hàm này sẽ được gọi trong quá trình forwarding của mạng

        Parameters
        ----------
        inputs: tensor đầu vào

        Returns
        -------
        tensor
            giá trị đầu ra của mạng
        '''
        ## TODO 3
        ### START CODE HERE
        x = self.cnn(inputs)
        ## END CODE HERE
        return x
class Alex_net(tf.keras.Model):
    def __init__(self, num_classes):
        super(Alex_net, self).__init__()
        ## TODO 4
        ### START CODE HERE
        self.conv1 =  ConvBlock(96,(11,11),4,'same')
        self.conv2 =  ConvBlock(256,(5,5),2,'same')
        self.conv3 =  ConvBlock(384,(3,3),1,'same')
        self.conv4 =  ConvBlock(384,(3,3),1,'same')
        self.conv5 =  ConvBlock(256,(3,3),1,'same')
        ### END CODE HERE
        ## TODO 5
        ### START CODE HERE
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation = 'relu')
        self.dropout1 = tf.keras.layers.Dropout(0.8)
        self.fc2 = tf.keras.layers.Dense(num_classes, activation = 'softmax')
        ### END CODE HERE

    def call(self, inputs):
        ## TODO 6
        x = inputs
        print(x.shape)
        ### START CODE HERE
        x = self.conv1(x)
        x= self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x=self.pool3(x)
        ### END CODE HERE
        ## TODO 7
        ### START CODE HERE
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        ### END CODE HERE
        return x