package learning.utils;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * The usual Counter
 */
public class Counter<V> implements Collection<V> {
  HashMap<V, Double> map;

  public Counter() {
    map = new HashMap<>();
  }
  public Counter(int initialCapacity) {
    map = new HashMap<>(initialCapacity);
  }

  @Override
  public int size() {
    return map.size();
  }

  @Override
  public boolean isEmpty() {
    return map.isEmpty();
  }

  @Override
  public boolean contains(Object o) {
    return map.containsKey(o);
  }

  @Override
  public Iterator<V> iterator() {
    return map.keySet().iterator();
  }

  @Override
  public Object[] toArray() {
    return map.keySet().toArray();
  }

  @Override
  public <T> T[] toArray(T[] a) {
    return map.keySet().toArray(a);
  }

  public boolean add(V item) {
    if(!map.containsKey(item))
      map.put(item, 1.0);
    else
      map.put(item, map.get(item) + 1.0);
    return true;
  }

  public boolean add(V item, Double value) {
    if(!map.containsKey(item))
      map.put(item, value);
    else
      map.put(item, map.get(item) + value);
    return true;
  }

  public double getCount(V item) {
    if( map.containsKey(item) )
      return map.get(item);
    else
      return 0.;
  }

  public double sum() {
    double total = 0.;
    for( Double count : map.values() ) {
      total += count;
    }
    return total;
  }

  @Override
  public boolean remove(Object o) {
    return map.remove(o) != null;
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    for( Object o : c ) {
      if(!map.containsKey(o)) return false;
    }
    return true;
  }

  @Override
  public boolean addAll(Collection<? extends V> c) {
    for( V o : c ) {
      add(o);
    }
    return c.size() > 0;
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    boolean removed = false;
    for( Object o : c ) {
      removed = removed || remove(o);
    }
    return removed;
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clear() {
    map.clear();
  }
}