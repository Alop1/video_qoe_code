import data_analyzer
import time

if __name__ == '__main__':

    start = time.time()
    for i in xrange(150):
        print "\niterarion no ", i
        data_analyzer.main()
    end = time.time()

    print "simulation durations: {}".format(end - start)