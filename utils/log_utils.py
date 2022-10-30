import codecs
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class RetrieverLogger:
    def __init__(self, data_path='', log_file='log.txt'):
        self.data_path = data_path
        self.log_file = log_file

        if self.data_path != '':
            with codecs.open('%s/%s' % (self.data_path, self.log_file), 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\taccuracy\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, accuracy, remark=''):
        with codecs.open('%s/%s' % (self.data_path, self.log_file), 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%.5f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, accuracy, remark))

    def draw_plot(self, data_path='', log_file=''):
        if data_path == '':
            data_path = self.data_path
        if log_file == '':
            log_file = self.log_file

        eppch = []
        loss = []
        accuracy = []
        with codecs.open('%s/%s' % (data_path, log_file), 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                loss.append(float(line[2]))
                accuracy.append(float(line[3]))

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((1, 2), (0, 0), title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((1, 2), (0, 1), title='accuracy')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, accuracy)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)


class ReaderLogger:
    def __init__(self, data_path='', log_file='log.txt'):
        self.data_path = data_path
        self.log_file = log_file

        if self.data_path != '':
            with codecs.open('%s/%s' % (self.data_path, self.log_file), 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\tf1\tem\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, f1, em, remark=''):
        with codecs.open('%s/%s' % (self.data_path, self.log_file), 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%.5f\t%.3f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, f1, em, remark))

    def draw_plot(self, data_path='', log_file=''):
        if data_path == '':
            data_path = self.data_path
        if log_file == '':
            log_file = self.log_file

        eppch = []
        loss = []
        f1 = []
        em = []

        with codecs.open('%s/%s' % (data_path, log_file), 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                loss.append(float(line[2]))
                f1.append(float(line[3]))
                em.append(float(line[4]))

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((2, 2), (0, 0), colspan=2, title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((2, 2), (1, 0), title='f1')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, f1)

        ax = plt.subplot2grid((2, 2), (1, 1), title='em')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, em)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)


class RankerLogger:
    def __init__(self, data_path='', log_file='log.txt'):
        self.data_path = data_path
        self.log_file = log_file

        if self.data_path != '':
            with codecs.open('%s/%s' % (self.data_path, self.log_file), 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\taccuracy\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, accuracy, remark=''):
        with codecs.open('%s/%s' % (self.data_path, self.log_file), 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%.5f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, accuracy, remark))

    def draw_plot(self, data_path='', log_file=''):
        if data_path == '':
            data_path = self.data_path
        if log_file == '':
            log_file = self.log_file

        eppch = []
        loss = []
        accuracy = []
        with codecs.open('%s/%s' % (data_path, log_file), 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                loss.append(float(line[2]))
                accuracy.append(float(line[3]))

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((1, 2), (0, 0), title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((1, 2), (0, 1), title='accuracy')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, accuracy)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)


class ValidatorLogger:
    def __init__(self, data_path='', log_file='log.txt'):
        self.data_path = data_path
        self.log_file = log_file

        if self.data_path != '':
            with codecs.open('%s/%s' % (self.data_path, self.log_file), 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\taccuracy\tprecision\trecall\tf1\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, accuracy, precision, recall, f1, remark=''):
        with codecs.open('%s/%s' % (self.data_path, self.log_file), 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, accuracy, precision, recall, f1, remark))

    def draw_plot(self, data_path='', log_file=''):
        if data_path == '':
            data_path = self.data_path
        if log_file == '':
            log_file = self.log_file

        eppch = []
        loss = []
        accuracy = []
        precision = []
        recall = []
        f1 = []

        with codecs.open('%s/%s' % (data_path, log_file), 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                loss.append(float(line[2]))
                accuracy.append(float(line[3]))
                precision.append(float(line[4]))
                recall.append(float(line[5]))
                f1.append(float(line[6]))

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((2, 3), (0, 0), title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((2, 3), (0, 1), title='accuracy')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, accuracy)

        ax = plt.subplot2grid((2, 3), (1, 0), title='precision')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, precision)

        ax = plt.subplot2grid((2, 3), (1, 1), title='recall')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, recall)

        ax = plt.subplot2grid((2, 3), (1, 2), title='f1')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, f1)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)
