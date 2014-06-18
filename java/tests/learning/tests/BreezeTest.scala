package learning.tests

import breeze.linalg._
import org.scalatest.FunSuite
import scala.collection.mutable.Stack
import scala.util.Random
import scala.collection.mutable
;

/**
 * Test the Breeze API
 */
class BreezeTest extends FunSuite {

  val rnd = new Random()

  test("pop is invoked on a non-empty stack") {

    val stack = new mutable.Stack[Int]
    stack.push(1)
    stack.push(2)
    val oldSize = stack.size
    val result =  stack.pop()
    assert(result === 2)
    assert(stack.size === oldSize - 1)
  }

  test("pop is invoked on an empty stack") {

    val emptyStack = new mutable.Stack[Int]
    intercept[NoSuchElementException] {
      emptyStack.pop()
    }
    assert(emptyStack.isEmpty)
  }

  test("vector dot products") {
    val x = DenseVector(1.0, 1.0, 1.0)
    val y = DenseVector(0.5, 0.5, 0.5)
    assert( x.dot(y) == 1.5 )
    assert( x.dot(y.:*(2.0)) == 2 )
  }

}
