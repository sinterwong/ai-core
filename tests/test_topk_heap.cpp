#include "topk_heap.hpp"
#include "gtest/gtest.h"

namespace topk_heap_test {

class TopKHeapTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TopKHeapTest, Normal) {

  common_utils::TopKHeap<int> heap(5);
  heap.push(2);
  heap.push(5);
  heap.push(1);
  heap.push(4);
  heap.push(6);
  ASSERT_EQ(heap.size(), 5);
  heap.push(3);
  ASSERT_EQ(heap.size(), 5);
  ASSERT_EQ(heap.top().value(), 2);
  auto topK = heap.getTopK();
  ASSERT_EQ(topK.size(), 5);
  std::vector<int> expectTopKA{6, 5, 4, 3, 2};
  ASSERT_EQ(topK, expectTopKA);
  heap.setCapacity(3);
  ASSERT_EQ(heap.capacity(), 3);
  topK = heap.getTopK();
  std::vector<int> expectTopKB{6, 5, 4};
  ASSERT_EQ(topK.size(), 3);
  ASSERT_EQ(topK, expectTopKB);

  common_utils::TopKHeap<int> heap2(heap);
  topK = heap2.getTopK();
  ASSERT_EQ(topK, expectTopKB);

  common_utils::TopKHeap<int> heap3;
  heap3 = heap;
  topK = heap3.getTopK();
  ASSERT_EQ(topK, expectTopKB);

  common_utils::TopKHeap<int> heap4(std::move(heap));
  topK = heap4.getTopK();
  ASSERT_EQ(topK, expectTopKB);

  common_utils::TopKHeap<int> heap5;
  heap5 = std::move(heap2);
  topK = heap5.getTopK();
  ASSERT_EQ(topK, expectTopKB);
}
} // namespace topk_heap_test
