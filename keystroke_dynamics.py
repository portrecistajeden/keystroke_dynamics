import json
import time
import numpy as np
import scipy as sp
from collections import Counter
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QWidget, QLabel, QTextEdit, QLineEdit, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, \
    QComboBox, QFileDialog, QMessageBox


class KeystrokeDynamics(QWidget):
    def __init__(self, parent=None):
        super(KeystrokeDynamics, self).__init__(parent)
        self.setWindowTitle('Keystroke Dynamics')
        self.tested_sentence = 'The quick brown fox jumps over a lazy dog'
        self.allowed_keys = [Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R, Qt.Key_T, Qt.Key_Y, Qt.Key_U, Qt.Key_I, Qt.Key_O,
                             Qt.Key_P, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_F, Qt.Key_G, Qt.Key_H, Qt.Key_J, Qt.Key_K,
                             Qt.Key_L, Qt.Key_Z, Qt.Key_X, Qt.Key_C, Qt.Key_V, Qt.Key_B, Qt.Key_N, Qt.Key_M]
        self.press_time = [-1] * len(self.allowed_keys)
        self.dwell_time = [None] * len(self.tested_sentence)
        self.index = 0
        self.pressed_keys = {}
        self.flight_time = []
        self.samples = []
        self.correct_classifications = 0
        self.all_classifications = 0

        self.label_username = QLabel('Your name: ')
        self.username = QLineEdit()

        self.button_load_samples = QPushButton('Load samples from file')
        self.button_save_samples = QPushButton('Save samples to file')
        self.label_number_of_samples = QLabel('Number of samples: ')
        self.number_of_samples = QLabel('0')
        self.calculating_method = QComboBox()
        self.button_calculate = QPushButton('Calculate')
        self.time = QComboBox()
        self.results = QTextEdit()
        self.correct_classifications_label = QLabel('Correct classifications: ')
        self.ccln = QLabel('0')
        self.quality_label = QLabel('Quality: ')
        self.quality_value = QLabel('0.0%')
        self.kValue = QLineEdit()

        self.text = QTextEdit()
        self.template = QLabel(self.tested_sentence)
        self.input = QLineEdit()
        self.button_reset = QPushButton('Reset')
        self.button_save = QPushButton('Save')
        self.quality_button = QPushButton('Examine quality')

        self.calculating_method.addItems(['Euklides', 'Manhattan', 'Czebyszew', 'Mahalanobis'])
        self.time.addItems(['Dwell', 'Flight'])
        self.results.setReadOnly(True)
        self.text.setReadOnly(True)
        self.input.setPlaceholderText('type here')
        self.input.installEventFilter(self)
        self.button_save.setEnabled(False)
        self.quality_button.setEnabled(False)
        self.button_calculate.setEnabled(False)
        self.kValue.setText('5')

        # layout creation
        samples = QHBoxLayout()
        samples.addWidget(self.label_number_of_samples)
        samples.addWidget(self.number_of_samples)

        classifications = QHBoxLayout()
        classifications.addWidget(self.correct_classifications_label)
        classifications.addWidget(self.ccln)

        quality = QHBoxLayout()
        quality.addWidget(self.quality_label)
        quality.addWidget(self.quality_value)

        time_and_k = QHBoxLayout()
        time_and_k.addWidget(self.time)
        time_and_k.addWidget(self.kValue)

        left_column = QVBoxLayout()
        left_column.addWidget(self.button_load_samples)
        left_column.addWidget(self.button_save_samples)
        left_column.addLayout(samples)
        left_column.addWidget(self.calculating_method)
        left_column.addLayout(time_and_k)
        left_column.addWidget(self.button_calculate)
        left_column.addWidget(self.results)
        left_column.addLayout(classifications)
        left_column.addLayout(quality)

        right_column = QGridLayout()
        right_column.addWidget(self.label_username, 0, 0)
        right_column.addWidget(self.username, 0, 1)
        right_column.addWidget(self.text, 1, 0, 1, 2)
        right_column.addWidget(self.template, 2, 0, 1, 2)
        right_column.addWidget(self.input, 3, 0, 1, 2)
        right_column.addWidget(self.button_reset, 4, 0)
        right_column.addWidget(self.button_save, 4, 1)
        right_column.addWidget(self.quality_button, 5, 0, 1, 2)

        # connecting signals
        self.button_load_samples.clicked.connect(self.load_from_file)
        self.username.textEdited.connect(self.check_if_valid)
        self.input.textEdited.connect(self.compare_input)
        self.input.textEdited.connect(self.check_if_valid)
        self.button_reset.clicked.connect(self.reset)
        self.button_save.clicked.connect(self.save)
        self.button_save_samples.clicked.connect(self.save_to_file)
        self.button_calculate.clicked.connect(self.calculate)
        self.quality_button.clicked.connect(self.badanko)

        layout = QHBoxLayout()
        layout.addLayout(left_column, 1)
        layout.addLayout(right_column, 2)
        self.setLayout(layout)
        self.show()

    def eventFilter(self, source, event):
        if event.type() in [QEvent.KeyPress, QEvent.KeyRelease]:
            if event.key() in self.allowed_keys:
                index = self.allowed_keys.index(event.key())
                if event.type() == QEvent.KeyPress:
                    press_time = time.time_ns() // 1000000
                    if not all(item == -1 for item in self.press_time):
                        self.flight_time.append(press_time - max(self.press_time))
                    self.press_time[index] = press_time
                    self.pressed_keys.update({event.key(): self.index})
                    self.index += 1
                    self.text.append('pressed {}'.format(event.text()))
                elif not all(item == -1 for item in self.press_time):
                    length = (time.time_ns() // 1000000) - self.press_time[index]
                    self.dwell_time[self.pressed_keys.pop(event.key())] = length
                    self.text.append('released {} after {} milliseconds'.format(event.text(), length))
        return super(KeystrokeDynamics, self).eventFilter(source, event)

    def compare_input(self, text):
        if not self.tested_sentence.lower().startswith(text.lower()):
            self.reset()

    def check_if_valid(self):
        text_len = len(self.input.text()) - self.input.text().count(' ')
        if self.input.text().lower() == self.tested_sentence.lower():
            if len(self.flight_time) != text_len - 1:
                raise IOError()
            if len(self.username.text()):
                self.button_save.setEnabled(True)
                self.button_calculate.setEnabled(True)
        else:
            self.button_save.setEnabled(False)
            self.button_calculate.setEnabled(False)

    def reset(self):
        self.press_time = [-1] * len(self.allowed_keys)
        self.dwell_time = [None] * len(self.tested_sentence)
        self.flight_time = []
        self.pressed_keys.clear()
        self.index = 0
        self.text.clear()
        self.input.clear()
        self.button_save.setEnabled(False)

    def save(self):
        self.samples.append({'user': self.username.text(),
                             'dwell_times': self.recalculate_dwell(),
                             'flight_times': self.flight_time})
        print(self.samples)
        self.number_of_samples.setText(str(len(self.samples)))
        self.reset()

    def load_from_file(self):
        loadFilePath = QFileDialog.getOpenFileName(self, 'Load file', './', "*.json")
        file = open(loadFilePath[0], 'r')
        self.samples = json.load(file)
        self.number_of_samples.setText(str(len(self.samples)))
        if len(self.samples) != 0:
            self.quality_button.setEnabled(True)

    def save_to_file(self):
        saveFilePath = QFileDialog.getSaveFileName(self, 'Save file', './', "*.json")
        file = open(saveFilePath[0], 'w')
        json.dump(self.samples, file)

    def pop_up_samples(self):
        QMessageBox.question(self, 'Error', "Not enough samples", QMessageBox.Ok)
        pass

    def calculate(self):
        self.results.setText('')
        current_sample = {'user': self.username.text(),
                          'dwell_times': self.recalculate_dwell(),
                          'flight_times': self.flight_time}
        matched_sample = self.calculate_deep(current_sample, self.samples)

        self.results.setText('Hmmm I think your name is: ' + matched_sample.get('user'))

    def calculate_deep(self, current_sample, samples):
        if len(samples) == 0:
            self.pop_up_samples()
            return

        calc_method = self.calculating_method.currentText()

        k_val = int(self.kValue.text())
        time = self.time.currentText()

        d_dict = []

        if calc_method == 'Euklides':
            for elem in samples:
                d_dict.append(self.Euklides(current_sample, elem))
        elif calc_method == 'Manhattan':
            for elem in samples:
                d_dict.append(self.Manhattan(current_sample, elem))
        elif calc_method == 'Czebyszew':
            for elem in samples:
                d_dict.append(self.Czebyszew(current_sample, elem))
        elif calc_method == 'Mahalanobis':
            for elem in samples:
                d_dict.append(self.Mahalanobis(current_sample, elem))

        d_dwell_list = sorted(d_dict, key=lambda x: x['d_dwell'], reverse=False)
        d_flight_list = sorted(d_dict, key=lambda x: x['d_flight'], reverse=False)
        d_dwell_list = d_dwell_list[:k_val]
        d_flight_list = d_flight_list[:k_val]

        if time == 'Dwell':
            matched_sample = self.find_matched_sample(d_dwell_list, k_val, time)
        else:
            matched_sample = self.find_matched_sample(d_flight_list, k_val, time)

        return matched_sample

    def find_matched_sample(self, d_list, kValue, time):
        pom_array = []
        user_array = []
        result = []
        sums = []
        for i in range(0, kValue):
            if d_list[i].get('user') not in pom_array:
                pom_array.append(d_list[i].get('user'))
            user_array.append(d_list[i].get('user'))

        for user in pom_array:
            result.append({'user': user,
                           'count': user_array.count(user)})

        result = sorted(result, key=lambda x: x['count'], reverse=True)

        if len(result) == 1:
            return result[0].get('user')
        if result[0].get('user') != result[1].get('user'):
            return result[0].get('user')
        else:
            sums = self.times_sums(d_list)
            if time == 'Dwell':
                sums = sorted(sums, key=lambda x: x['sum_d'], reverse=False)
            else:
                sums = sorted(sums, key=lambda x: x['sum_f'], reverse=False)
        return sums[0].get('user')
                    
    def times_sums(self, dict):
        times_d = dict.get('dwell_times')
        times_f = dict.get('flight_times')
        sum_d = 0
        sum_f = 0
        result = []
        for i in range(0, len(times_d)):
            sum_d += times_d[i]
        for i in range(0, len(times_f)):
            sum_f += times_f[i]
        result = {'user': dict.get('user'),
                  'sum_d': sum_d,
                  'sum_f': sum_f}
        return result


    def Euklides(self, dict_current, dict_from_db):
        dwell_times_curr = dict_current.get('dwell_times')
        flight_times_curr = dict_current.get('flight_times')

        dwell_times_fromDB = dict_from_db.get('dwell_times')
        flight_times_fromDb = dict_from_db.get('flight_times')

        sum_dwell = 0
        sum_flight = 0

        for i in range(0, len(dwell_times_curr)):
            sum_dwell += np.power(dwell_times_curr[i] - dwell_times_fromDB[i], 2)

        for i in range(0, len(flight_times_curr)):
            sum_flight += np.power(flight_times_curr[i] - flight_times_fromDb[i], 2)

        result = {'user': dict_from_db.get('user'),
                  'd_dwell': np.sqrt(sum_dwell),
                  'd_flight': np.sqrt(sum_flight)}

        return result

    def Mahalanobis(self, dict_current, dict_from_db):
        dwell_times_curr = dict_current.get('dwell_times')
        flight_times_curr = dict_current.get('flight_times')

        dwell_times_fromDB = dict_from_db.get('dwell_times')
        flight_times_fromDb = dict_from_db.get('flight_times')

        normal_sum_d = 0
        normal_sum_f = 0

        identity_matrix = np.identity(len(dwell_times_curr))
        VI = np.linalg.inv(np.cov(identity_matrix))
        result_d_dwell = sp.mahalanobis(dwell_times_curr, dwell_times_fromDB, VI)
        identity_matrix_2 = np.identity(len(flight_times_curr))
        VI_2 = np.linalg.inv(np.cov(identity_matrix_2))
        result_d_flight = sp.mahalanobis(flight_times_curr, flight_times_fromDb, VI_2)

        for i in range(0, len(dwell_times_curr)):
            normal_sum_d += dwell_times_fromDB[i]
        for i in range(0, len(flight_times_curr)):
            normal_sum_f += flight_times_fromDb[i]

        result = {'user': dict_from_db.get('user'),
                  'd_dwell': result_d_dwell,
                  'dwell_sum': normal_sum_d,
                  'd_flight': result_d_flight,
                  'flight_sum': normal_sum_f}

        return result

    def Czebyszew(self, dict_current, dict_from_db):
        dwell_times_curr = dict_current.get('dwell_times')
        flight_times_curr = dict_current.get('flight_times')

        dwell_times_fromDB = dict_from_db.get('dwell_times')
        flight_times_fromDb = dict_from_db.get('flight_times')

        absolute_sums_dwell = []
        absolute_sums_flight = []

        for i in range(0, len(dwell_times_curr)):
            absolute_sums_dwell.append(dwell_times_curr[i] - dwell_times_fromDB[i])

        for i in range(0, len(flight_times_curr)):
            absolute_sums_flight.append(flight_times_curr[i] - flight_times_fromDb[i])

        result_d_dwell = np.amax(absolute_sums_dwell)
        result_d_flight = np.amax(absolute_sums_flight)

        result = {'user': dict_from_db.get('user'),
                  'd_dwell': result_d_dwell,
                  'd_flight': result_d_flight}

        return result

    def Mahalanobis(self, dict_current, dict_from_db):
        dwell_times_curr = dict_current.get('dwell_times')
        flight_times_curr = dict_current.get('flight_times')

        dwell_times_fromDB = dict_from_db.get('dwell_times')
        flight_times_fromDb = dict_from_db.get('flight_times')


        identity_matrix = np.ones(len(dwell_times_curr))
        VI = np.linalg.inv(np.cov(identity_matrix))
        result_d_dwell = sp.spatial.distance.mahalanobis(dwell_times_curr, dwell_times_fromDB, VI)
        result_d_flight = sp.spatial.distance.mahalanobis(flight_times_curr, flight_times_fromDb, VI)

        result = {'user': dict_from_db.get('user'),
                  'd_dwell': result_d_dwell,
                  'd_flight': result_d_flight}

        return result

    def recalculate_dwell(self):
        sentence = self.tested_sentence.replace(" ", "")
        sentence = sentence.lower()
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        new_dwell = []
        for letter in alphabet:
            total = 0
            number_of_occurrences = 0
            for i, value in enumerate(sentence):
                if value == letter:
                    total += self.dwell_time[i]
                    number_of_occurrences += 1
            if number_of_occurrences == 0:
                new_dwell.append(0)
            else:
                new_dwell.append(total/number_of_occurrences)
        return new_dwell

    def examine_quality(self):
        self.quality_value.setText(format(self.correct_classifications/self.all_classifications*100, '.2f') + '%')

    def badanko(self):
        self.all_classifications = 0
        self.correct_classifications = 0
        for sample in self.samples:
            rest_samples = self.samples.copy()
            rest_samples.remove(sample)
            matched_sample = self.calculate_deep(sample, rest_samples)
            self.all_classifications += 1
            if sample.get('user') == matched_sample:
                self.correct_classifications += 1

        self.examine_quality()
        self.ccln.setText(str(self.correct_classifications) + '/' + str(self.all_classifications))
