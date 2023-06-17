    #%% md
2    Loading in packages and pre set to use on the toy data set
3    #%%
4    import h5py
5    import numpy as np
6
7    hf = h5py.File('toy_dataset.h5', 'r')
8    hf.keys()
9    ecgs = np.array(hf.get('ECGs'))
10   labels = np.array(hf.get('labels'))
11
12
13   #%% md
14   Visualising the 12 channels of the first ecg
15   #%%
16   import matplotlib.pyplot as plt
17
18   # Assuming 'ecgs' is the variable containing the ECG data
19   first_ecg = ecgs[0]  # Get the first ECG
20
21   # Plotting the 12 channels of the first ECG
22   for i in range(12):
23       plt.subplot(4, 3, i+1)
24       plt.plot(first_ecg[:, i])
25       plt.title('Channel {}'.format(i+1))
26
27   plt.tight_layout()
28   plt.show()
29
30   #%% md
31   Visualing the leads the way David did it because he probably understands ECG a bit better than me
32   #%%
33   first_ecg = ecgs[0]
34   fig, axs = plt.subplots(nrows=12, sharex=True, figsize=(10, 15))
35   for i in range(12):
36       axs[i].plot(first_ecg[:, i])
37       axs[i].set_ylabel(f'Lead {i+1}')
38   axs[-1].set_xlabel('Time (s)')
39   axs[-1].set_ylabel('Amplitude')
40   plt.tight_layout()
41   plt.show()
42   #%% md
43   using a simple CNN on the ECG dataset to test if it works
44   #%%
45   import numpy as np
46   import tensorflow as tf
47   from tensorflow.keras.models import Sequential
48   from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
49   from sklearn.model_selection import train_test_split
50   from sklearn.metrics import confusion_matrix
51
52   # Assuming 'ecgs' is the variable containing the ECG data
53   # Assuming 'labels' is the variable containing the corresponding labels
54
55   # Split the data into train, validation, and test sets
56   X_train_val, X_test, y_train_val, y_test = train_test_split(ecgs, labels, test_size=0.2, random_state=42)
57   X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
58
59   # Define the CNN model architecture
60   model = Sequential()
61   model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
62   model.add(MaxPooling1D(pool_size=2))
63   model.add(Flatten())
64   model.add(Dense(128, activation='relu'))
65   model.add(Dense(1, activation='sigmoid'))
66
67   # Compile the model
68   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
69
70   # Train the model on the training set and validate on the validation set
71   model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))
72
73
74
75   #%% md
76   Analysis the effectiveness of the model using test data set and confusion matrix
77   #%%
78   import numpy as np
79   import matplotlib.pyplot as plt
80   import itertools
81   from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
82
83   # Making predictions on the test set
84   y_pred = model.predict(X_test)
85   y_pred_classes = (y_pred > 0.5).astype(np.int)
86   y_true_classes = y_test.astype(np.int)
87
88   # Calculate the confusion matrix
89   cm = confusion_matrix(y_true_classes, y_pred_classes)
90
91   # Calculate sensitivity and specificity for the positive class (class 1)
92   true_positives = cm[1, 1]
93   false_positives = cm[0, 1]
94   false_negatives = cm[1, 0]
95   true_negatives = cm[0, 0]
96
97   sensitivity = true_positives / (true_positives + false_negatives)
98   specificity = true_negatives / (true_negatives + false_positives)
99
100  # Calculate overall accuracy, F1-score
101  overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)
102  overall_f1_score = f1_score(y_true_classes, y_pred_classes)
103
104  # Print overall metrics
105  print(f'Overall Accuracy: {overall_accuracy:.2f}')
106  print(f'Sensitivity (True Positive Rate): {sensitivity:.2f}')
107  print(f'Specificity (True Negative Rate): {specificity:.2f}')
108  print(f'Overall F1-score: {overall_f1_score:.2f}')
109
110  # Plot ROC curve
111  fpr, tpr, _ = roc_curve(y_true_classes, y_pred)
112  roc_auc = roc_auc_score(y_true_classes, y_pred)
113  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
114  plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
115
116  plt.xlim([0.0, 1.0])
117  plt.ylim([0.0, 1.0])
118  plt.xlabel('False Positive Rate')
119  plt.ylabel('True Positive Rate')
120  plt.title('ROC Curve for Binary Classification')
121  plt.legend(loc="lower right")
122
123  plt.show()
124
125  # Plot confusion matrix
126  plt.imshow(cm, cmap=plt.cm.Blues)
127  plt.title('Confusion Matrix for Binary Classification')
128  plt.colorbar()
129
130  class_names = ['Class 0', 'Class 1']  # Modify with your class names
131  tick_marks = np.arange(len(class_names))
132  plt.xticks(tick_marks, class_names, rotation=45)
133  plt.yticks(tick_marks, class_names)
134
135  fmt = 'd'
136  thresh = cm.max() / 2.
137  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
138      plt.text(j, i, format(cm[i, j], fmt),
139               horizontalalignment="center",
140               color="white" if cm[i, j] > thresh else "black")
141
142  plt.ylabel('True label')
143  plt.xlabel('Predicted label')
144  plt.tight_layout()
145
146  plt.show()
147
148  #%% md
149  Adding additional conv1d layers to see if the model can pick anything else up
150  #%%
151  import numpy as np
152  import tensorflow as tf
153  from tensorflow.keras.models import Sequential
154  from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
155  from sklearn.model_selection import train_test_split
156  from sklearn.metrics import confusion_matrix
157
158
159
160  # Splitting the data into train, validation, and test sets
161  X_train_val, X_test, y_train_val, y_test = train_test_split(ecgs, labels, test_size=0.2, random_state=42)
162  X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
163  # Define the CNN model architecture
164  model = Sequential()
165  model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
166  model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
167  model.add(MaxPooling1D(pool_size=2))
168  model.add(Dropout(0.2))
169  model.add(Flatten())
170  model.add(Dense(128, activation='relu'))
171  model.add(Dropout(0.5))
172  model.add(Dense(1, activation='sigmoid'))
173
174  # Compile the model
175  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
176
177  # Train the model on the training set and validate on the validation set
178  model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))
179
180  #%%
181  import numpy as np
182  import matplotlib.pyplot as plt
183  import itertools
184  from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
185
186  # Making predictions on the test set
187  y_pred = model.predict(X_test)
188  y_pred_classes = (y_pred > 0.5).astype(np.int)
189  y_true_classes = y_test.astype(np.int)
190
191  # Calculate the confusion matrix
192  cm = confusion_matrix(y_true_classes, y_pred_classes)
193
194  # Calculate sensitivity and specificity for the positive class (class 1)
195  true_positives = cm[1, 1]
196  false_positives = cm[0, 1]
197  false_negatives = cm[1, 0]
198  true_negatives = cm[0, 0]
199
200  sensitivity = true_positives / (true_positives + false_negatives)
201  specificity = true_negatives / (true_negatives + false_positives)
202
203  # Calculate overall accuracy, F1-score
204  overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)
205  overall_f1_score = f1_score(y_true_classes, y_pred_classes)
206
207  # Print overall metrics
208  print(f'Overall Accuracy: {overall_accuracy:.2f}')
209  print(f'Sensitivity (True Positive Rate): {sensitivity:.2f}')
210  print(f'Specificity (True Negative Rate): {specificity:.2f}')
211  print(f'Overall F1-score: {overall_f1_score:.2f}')
212
213  # Plot ROC curve
214  fpr, tpr, _ = roc_curve(y_true_classes, y_pred)
215  roc_auc = roc_auc_score(y_true_classes, y_pred)
216  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
217  plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
218
219  plt.xlim([0.0, 1.0])
220  plt.ylim([0.0, 1.0])
221  plt.xlabel('False Positive Rate')
222  plt.ylabel('True Positive Rate')
223  plt.title('ROC Curve for Binary Classification')
224  plt.legend(loc="lower right")
225
226  plt.show()
227
228  # Plot confusion matrix
229  plt.imshow(cm, cmap=plt.cm.Blues)
230  plt.title('Confusion Matrix for Binary Classification')
231  plt.colorbar()
232
233  class_names = ['Class 0', 'Class 1']  # Modify with your class names
234  tick_marks = np.arange(len(class_names))
235  plt.xticks(tick_marks, class_names, rotation=45)
236  plt.yticks(tick_marks, class_names)
237
238  fmt = 'd'
239  thresh = cm.max() / 2.
240  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
241      plt.text(j, i, format(cm[i, j], fmt),
242               horizontalalignment="center",
243               color="white" if cm[i, j] > thresh else "black")
244
245  plt.ylabel('True label')
246  plt.xlabel('Predicted label')
247  plt.tight_layout()
248
249  plt.show()
250  #%% md
251  Overall using more convolutional layers way not very helpful probably because there is not much more detail in the first place to need more conolutional layers
252  #%% md
253  Using some of the implementation mentioned in slack to do the analysis
254  This is the model that was mentioned in the first paper
255
256
257  #%%
258  import numpy as np
259  import tensorflow as tf
260  from tensorflow.keras.models import Sequential, Model
261  from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, ZeroPadding1D, Input, BatchNormalization, Activation
262  from tensorflow.keras.initializers import glorot_uniform
263  from sklearn.model_selection import train_test_split
264  from sklearn.metrics import confusion_matrix
265
266  def convolutional_block(X, f, filters, maxpool):
267      X = Conv1D(filters=filters, kernel_size=(f), strides=(1), padding='same',
268                 kernel_initializer=glorot_uniform(seed=0))(X)
269      X = BatchNormalization(axis=-1)(X)
270      X = Activation('relu')(X)
271      X = MaxPooling1D(maxpool, strides=None, padding='same')(X)
272      return X
273
274  def define_model(input_shape, classes, kernel_initializer=glorot_uniform(seed=0), drop = 0.5):
275
276      # Stage 1
277      X_input = Input(input_shape)
278      X = ZeroPadding1D((24))(X_input) #remove if input data already zero padded
279
280      # Stage 2
281      X = convolutional_block(X, f=5, filters=16, maxpool = 2)
282      X = convolutional_block(X, f=5, filters=16, maxpool = 2)
283      X = convolutional_block(X, f=5, filters=32, maxpool = 4)
284      X = convolutional_block(X, f=3, filters=32, maxpool = 2)
285      X = convolutional_block(X, f=3, filters=64, maxpool = 2)
286      X = convolutional_block(X, f=3, filters=64, maxpool = 4)
287
288      # Stage 3
289      X = Conv1D(filters=128, kernel_size=(3), strides=(1), padding='same',
290                 kernel_initializer=kernel_initializer)(X)
291      X = BatchNormalization(axis=-1)(X)
292      X = Activation('relu')(X)
293
294      # Stage 4
295      X = Flatten(name='flatten')(X)
296      X = Dense(units = 64)(X)
297      X = BatchNormalization(axis=-1)(X)
298      X = Activation('relu')(X)
299      if drop >0:
300          X = Dropout(drop)(X)
301      X = Dense(units = 32)(X)
302      X = BatchNormalization(axis=-1)(X)
303      X = Activation('relu')(X)
304      if drop >0:
305          X = Dropout(drop)(X)
306
307      # Output layer
308      X = Dense(classes, activation='sigmoid')(X)
309
310      # Create model
311      model = Model(inputs=X_input, outputs=X, name='cnn_ecg')
312      return model
313  #%%
314
315
316  # Reshaping y_train and y_val
317  y_train = np.reshape(y_train, (-1, 1))
318  y_val = np.reshape(y_val, (-1, 1))
319
320  # Defining input shape and number of classes
321  input_shape = X_train.shape[1:]
322  num_classes = 1  # For binary classification, set num_classes to 1
323
324  # Create the model using the define model function mentioned above
325  model = define_model(input_shape, num_classes)
326
327  # Compile the model on the new function
328  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
329
330  import matplotlib.pyplot as plt
331
332  # Train the model and store the history
333  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
334
335  # Access the loss values from the history
336  train_loss = history.history['loss']
337  val_loss = history.history['val_loss']
338
339  # Create a plot of training and validation losses
340  epochs = range(1, len(train_loss) + 1)
341  plt.plot(epochs, train_loss, 'b', label='Training Loss')
342  plt.plot(epochs, val_loss, 'r', label='Validation Loss')
343  plt.title('Training and Validation Loss')
344  plt.xlabel('Epochs')
345  plt.ylabel('Loss')
346  plt.legend()
347  plt.show()
348
349  # Evaluate the model on the test set
350  loss, accuracy = model.evaluate(X_test, y_test)
351  print(f'Test Loss: {loss:.2f}')
352  print(f'Test Accuracy: {accuracy:.2f}')
353
354
355  #%% md
356  analysing the effectivness of this new model
357  #%%
358  import numpy as np
359  import matplotlib.pyplot as plt
360  import itertools
361  from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report
362
363
364  y_pred = model.predict(X_test)
365  y_pred_classes = (y_pred > 0.5).astype(np.int)
366  y_true_classes = y_test.astype(np.int)
367
368  # Calculate the confusion matrix
369  cm = confusion_matrix(y_true_classes, y_pred_classes)
370
371  # Calculate sensitivity and specificity for the positive class (class 1)
372  true_positives = cm[1, 1]
373  false_positives = cm[0, 1]
374  false_negatives = cm[1, 0]
375  true_negatives = cm[0, 0]
376
377  sensitivity = true_positives / (true_positives + false_negatives)
378  specificity = true_negatives / (true_negatives + false_positives)
379
380  # Calculate overall accuracy, F1-score
381  overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)
382  overall_f1_score = f1_score(y_true_classes, y_pred_classes)
383
384  # Print overall metrics
385  print(f'Overall Accuracy: {overall_accuracy:.2f}')
386  print(f'Sensitivity (True Positive Rate): {sensitivity:.2f}')
387  print(f'Specificity (True Negative Rate): {specificity:.2f}')
388  print(f'Overall F1-score: {overall_f1_score:.2f}')
389
390  # Generate classification report
391  report = classification_report(y_true_classes, y_pred_classes, target_names=['Class 0', 'Class 1'])
392  print(report)
393
394  # Plot ROC curve
395  fpr, tpr, _ = roc_curve(y_true_classes, y_pred)
396  roc_auc = roc_auc_score(y_true_classes, y_pred)
397  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
398  plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
399
400  plt.xlim([0.0, 1.0])
401  plt.ylim([0.0, 1.0])
402  plt.xlabel('False Positive Rate')
403  plt.ylabel('True Positive Rate')
404  plt.title('ROC Curve for Binary Classification')
405  plt.legend(loc="lower right")
406
407  plt.show()
408
409  # Plot confusion matrix
410  plt.imshow(cm, cmap=plt.cm.Blues)
411  plt.title('Confusion Matrix for Binary Classification')
412  plt.colorbar()
413
414  class_names = ['Class 0', 'Class 1']  # Modify with your class names
415  tick_marks = np.arange(len(class_names))
416  plt.xticks(tick_marks, class_names, rotation=45)
417  plt.yticks(tick_marks, class_names)
418
419  fmt = 'd'
420  thresh = cm.max() / 2.
421  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
422      plt.text(j, i, format(cm[i, j], fmt),
423               horizontalalignment="center",
424               color="white" if cm[i, j] > thresh else "black")
425
426  plt.ylabel('True label')
427  plt.xlabel('Predicted label')
428  plt.tight_layout()
429
430  plt.show()
431  #%% md
432  Attempting to use RESNET with model
433  #%% md
434  This RESNET50 is used from the slack
435  #%%
436  'Resnet'
437  class ResidualUnit(object):
438
439      def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
440                   dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
441                   postactivation_bn=False, activation_function='relu'):
442          self.n_samples_out = n_samples_out
443          self.n_filters_out = n_filters_out
444          self.kernel_initializer = kernel_initializer
445          self.dropout_rate = 1 - dropout_keep_prob
446          self.kernel_size = kernel_size
447          self.preactivation = preactivation
448          self.postactivation_bn = postactivation_bn
449          self.activation_function = activation_function
450
451      def _skip_connection(self, y, downsample, n_filters_in):
452          """Implement skip connection."""
453          # Deal with downsampling
454          if downsample > 1:
455              y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
456          elif downsample == 1:
457              y = y
458          else:
459              raise ValueError("Number of samples should always decrease.")
460          # Deal with n_filters dimension increase
461          if n_filters_in != self.n_filters_out:
462              # This is one of the two alternatives presented in ResNet paper
463              # Other option is to just fill the matrix with zeros.
464              y = Conv1D(self.n_filters_out, 1, padding='same',
465                         use_bias=False, kernel_initializer=self.kernel_initializer)(y)
466          return y
467
468      def _batch_norm_plus_activation(self, x):
469          if self.postactivation_bn:
470              x = Activation(self.activation_function)(x)
471              x = BatchNormalization(center=False, scale=False)(x)
472          else:
473              x = BatchNormalization()(x)
474              x = Activation(self.activation_function)(x)
475          return x
476
477      def __call__(self, inputs):
478          """Residual unit."""
479          x, y = inputs
480          n_samples_in = y.shape[1]
481          downsample = n_samples_in // self.n_samples_out
482          n_filters_in = y.shape[2]
483          y = self._skip_connection(y, downsample, n_filters_in)
484          # 1st layer
485          x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
486                     use_bias=False, kernel_initializer=self.kernel_initializer)(x)
487          x = self._batch_norm_plus_activation(x)
488          if self.dropout_rate > 0:
489              x = Dropout(self.dropout_rate)(x)
490
491          # 2nd layer
492          x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
493                     padding='same', use_bias=False,
494                     kernel_initializer=self.kernel_initializer)(x)
495          if self.preactivation:
496              x = Add()([x, y])  # Sum skip connection and main connection
497              y = x
498              x = self._batch_norm_plus_activation(x)
499              if self.dropout_rate > 0:
500                  x = Dropout(self.dropout_rate)(x)
501          else:
502              x = BatchNormalization()(x)
503              x = Add()([x, y])  # Sum skip connection and main connection
504              x = Activation(self.activation_function)(x)
505              if self.dropout_rate > 0:
506                  x = Dropout(self.dropout_rate)(x)
507              y = x
508          return [x, y]
509
510
511  def get_model(n_classes, last_layer='sigmoid'):
512      kernel_size = 16
513      kernel_initializer = 'he_normal'
514      signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
515      x = signal
516      x = Conv1D(64, kernel_size, padding='same', use_bias=False,
517                 kernel_initializer=kernel_initializer)(x)
518      x = BatchNormalization()(x)
519      x = Activation('relu')(x)
520      x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
521                          kernel_initializer=kernel_initializer)([x, x])
522      x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
523                          kernel_initializer=kernel_initializer)([x, y])
524      x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
525                          kernel_initializer=kernel_initializer)([x, y])
526      x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
527                          kernel_initializer=kernel_initializer)([x, y])
528      x = Flatten()(x)
529      diagn = Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)(x)
530      model = Model(signal, diagn)
531      return model
532  #%% md
533  Training the resnet model
534  #%%
535  from tensorflow.keras.models import Model
536  from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Add, Flatten, Dense
537  from sklearn.model_selection import train_test_split
538  import numpy as np
539
540  # Reshape labels
541  y_train = np.reshape(y_train, (-1, 1))
542  y_val = np.reshape(y_val, (-1, 1))
543  y_test = np.reshape(y_test, (-1, 1))
544  # Create the model
545  model = get_model(2)
546
547  # Compile the model
548  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
549
550  import matplotlib.pyplot as plt
551
552  # Train the model and store the history
553  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
554
555  # Access the loss values from the history
556  train_loss = history.history['loss']
557  val_loss = history.history['val_loss']
558
559  # Create a plot of training and validation losses
560  epochs = range(1, len(train_loss) + 1)
561  plt.plot(epochs, train_loss, 'b', label='Training Loss')
562  plt.plot(epochs, val_loss, 'r', label='Validation Loss')
563  plt.title('Training and Validation Loss')
564  plt.xlabel('Epochs')
565  plt.ylabel('Loss')
566  plt.legend()
567  plt.show()
568
569  # Evaluate the model on the test set
570  loss, accuracy = model.evaluate(X_test, y_test)
571  print("Test Loss:", loss)
572  print("Test Accuracy:", accuracy)
573
574
575  #%% md
576  plotting confusion matrix of the model
577  #%%
578  import numpy as np
579  import matplotlib.pyplot as plt
580  import itertools
581  from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report
582
583
584  #prediction on the test dataset
585  y_pred = model.predict(X_test)
586  y_pred_classes = (y_pred > 0.5).astype(np.int)
587  y_true_classes = y_test.astype(np.int)
588
589  # Calculate the confusion matrix
590  cm = confusion_matrix(y_true_classes, y_pred_classes)
591
592  # Calculate sensitivity and specificity for the positive class (class 1)
593  true_positives = cm[1, 1]
594  false_positives = cm[0, 1]
595  false_negatives = cm[1, 0]
596  true_negatives = cm[0, 0]
597
598  sensitivity = true_positives / (true_positives + false_negatives)
599  specificity = true_negatives / (true_negatives + false_positives)
600
601  # Calculate overall accuracy, F1-score
602  overall_accuracy = accuracy_score(y_true_classes, y_pred_classes)
603  overall_f1_score = f1_score(y_true_classes, y_pred_classes)
604
605  # Print overall metrics
606  print(f'Overall Accuracy: {overall_accuracy:.2f}')
607  print(f'Sensitivity (True Positive Rate): {sensitivity:.2f}')
608  print(f'Specificity (True Negative Rate): {specificity:.2f}')
609  print(f'Overall F1-score: {overall_f1_score:.2f}')
610
611  # Generate classification report
612  report = classification_report(y_true_classes, y_pred_classes, target_names=['Class 0', 'Class 1'])
613  print(report)
614
615  # Plot ROC curve
616  fpr, tpr, _ = roc_curve(y_true_classes, y_pred)
617  roc_auc = roc_auc_score(y_true_classes, y_pred)
618  plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
619  plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
620
621  plt.xlim([0.0, 1.0])
622  plt.ylim([0.0, 1.0])
623  plt.xlabel('False Positive Rate')
624  plt.ylabel('True Positive Rate')
625  plt.title('ROC Curve for Binary Classification')
626  plt.legend(loc="lower right")
627
628  plt.show()
629
630  # Plot confusion matrix
631  plt.imshow(cm, cmap=plt.cm.Blues)
632  plt.title('Confusion Matrix for Binary Classification')
633  plt.colorbar()
634
635  class_names = ['Class 0', 'Class 1']  # Modify with your class names
636  tick_marks = np.arange(len(class_names))
637  plt.xticks(tick_marks, class_names, rotation=45)
638  plt.yticks(tick_marks, class_names)
639
640  fmt = 'd'
641  thresh = cm.max() / 2.
642  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
643      plt.text(j, i, format(cm[i, j], fmt),
644               horizontalalignment="center",
645               color="white" if cm[i, j] > thresh else "black")

647  plt.ylabel('True label')
plt.xlabel('Predicted label')
  plt.tight_layout()

 plt.show()