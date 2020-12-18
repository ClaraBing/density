import cProfile, pstats, io
from pstats import SortKey

def print_profile_stats(pr, fout, key=SortKey.CUMULATIVE):
  f = open(fout, 'w')
  ps = pstats.Stats(pr, stream=f).sort_stats(key)
  ps.print_stats()
