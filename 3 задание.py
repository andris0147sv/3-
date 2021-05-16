import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class Gaussian:
    ''' Класс с уравнением плоской волны для гауссова сигнала в дискретном виде
    d - определяет задержку сигнала.
    w - определяет ширину сигнала.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''
    def __init__(self, d, w, eps=1.0, mu=1.0, Sc=1.0,):
        self.d = d
        self.w = w
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getField(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return numpy.exp(-(((q - m * numpy.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2)


if __name__ == '__main__':
    W0 = 120.0 * numpy.pi
    Sc = 1.0
    c = 3e8
    maxTime = 1200
    X = 0.5
    dx = 1e-3
    maxSize = int(X / dx)
    dt = Sc * dx / c
    sourcePos = int(maxSize / 2)


    probesPos = [sourcePos + 150]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    eps = numpy.ones(maxSize)
    eps[:] = 2.0

    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = Gaussian(30.0, 7.0, eps[sourcePos], mu[sourcePos])

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]
    
    # Расчет коэффициентов для граничных условий
    tempRight = Sc / numpy.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1


    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)


        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия слева (PEC)
        Ez[0] = 0.0

        # Граничные условия ABC первой степени (справа)
        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()

    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    '''
    Спектр гауссова сигнала
    '''

    size = 2 ** 12


    z = fft(probe.E, size)
    z = fftshift(z)
    

    df = 1.0 / (size * dt)
    freq = numpy.arange(-size / 2 * df, size / 2 * df, df)


    plt.plot(freq * 1e-9, abs(z / numpy.max(z)))
    plt.xlim(0, 40e9 * 1e-9)
    plt.xlabel('Частота, ГГц')
    plt.ylabel('|S / Smax|')
    plt.grid()
    plt.show()
