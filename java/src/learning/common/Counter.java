package learning.common;

import java.util.*;

/**
 * The usual Counter
 */
public class Counter<V> implements Collection<V> {
  HashMap<V, Double> map;
  public double sum = 0.;

  public Counter() {
    map = new HashMap<>();
  }
  public Counter(int initialCapacity) {
    map = new HashMap<>(initialCapacity);
  }
  public Counter(Collection<V> initial) {
    this(initial.size());
    for(V elem : initial) {
      add(elem);
    }
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

  public void set(V item, Double value) {
    if(map.containsKey(item)) {
      sum -= map.get(item);
    }
    map.put(item, value);
    sum += value;
  }

  public boolean add(V item) {
    if(!map.containsKey(item))
      map.put(item, 1.0);
    else
      map.put(item, map.get(item) + 1.0);
    sum += 1.0;
    return true;
  }

  public boolean add(V item, Double value) {
    if(!map.containsKey(item))
      map.put(item, value);
    else
      map.put(item, map.get(item) + value);
    sum += value;
    return true;
  }

  public double getCount(V item) {
    if( map.containsKey(item) )
      return map.get(item);
    else
      return 0.;
  }

  public double getFraction(V item) {
    return getCount(item)/sum();
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
    Double count = map.remove(o);
    sum -= (count != null) ? count : 0.;
    return count != null;
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

  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("{");
    for( Map.Entry<V,Double> item : map.entrySet() )
      builder.append(item.getKey()).append(": ").append(item.getValue()).append(", ");
    builder.append("}");
    return builder.toString();
  }

  public static <V> double diff(Counter<V> e1, Counter<V> e2) {
    Set<V> keys = new HashSet<V>();
    keys.addAll(e1.map.keySet());
    keys.addAll(e2.map.keySet());

    double diff = 0.;

    for(V key : keys) {
      diff += Math.pow(e1.getCount(key) - e2.getCount(key), 2);
    }

    return diff;
  }

}
