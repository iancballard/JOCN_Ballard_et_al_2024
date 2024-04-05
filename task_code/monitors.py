"""Hold information about different monitors."""
from textwrap import dedent


bic = dict(name='BIC',
              width=24,
              distance=61,
              size=[1024, 768],
              framerate = 60,
              notes=dedent('''BIC scanner. Width measured by measuring projected image
              in scanner bore. Monitor set to tape 4. Distance measured assuming a 7cm
              distance from the mirror to subject eyes and a 54cm distance to tape 4'''))

mbpro = dict(name='MBpro',
                    width=33,
                    size=[2560, 1600],
                    distance=45,
                    framerate = 60,
                    notes="Ian Ballard's 13inch MBpro.")
