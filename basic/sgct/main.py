# Combination modules
from combinationScheme import combinationSchemeArbitrary
import ActiveSetFactory

lmin = (2,2)
lmax = (7,7)
factory = ActiveSetFactory.ClassicDiagonalActiveSet(lmax, lmin, 0)
activeSet = factory.getActiveSet()
scheme = combinationSchemeArbitrary(activeSet)
print scheme.dictOfScheme
