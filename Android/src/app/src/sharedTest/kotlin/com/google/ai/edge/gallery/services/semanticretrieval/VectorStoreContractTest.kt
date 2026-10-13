/*
 * Copyright 2026 ObjectBox Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.services.semanticretrieval

import com.google.mediapipe.tasks.core.TextPart
import com.google.mediapipe.tasks.retrieval.components.VectorStore
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test

/**
 * Behavior every [VectorStore] used by [SemanticRetrievalService] must have.
 *
 * Subclasses provide the implementation under test. The expectations describe how MediaPipe's
 * `SqliteVectorStore` behaves today, which is the baseline a replacement store has to match.
 */
abstract class VectorStoreContractTest {
  /** Creates a store backed by storage named [name]; a new [name] must give an empty store. */
  protected abstract fun createStore(name: String): VectorStore

  /** Whether closing a store and creating it again with the same name keeps its records. */
  protected open val persistsAcrossReopen: Boolean = false

  /**
   * Whether [VectorStore.upsert] with an existing id replaces that record. The app relies on
   * replacement (see [SemanticRetrievalServiceContractTest]); only stores that get it from a layer
   * above them, like `SqliteVectorStore` under `SemanticRetriever`, may opt out.
   */
  protected open val upsertReplacesExistingId: Boolean = true

  private lateinit var storeName: String
  protected lateinit var store: VectorStore

  @Before
  fun setUpStore() {
    storeName = "contract-${javaClass.simpleName}-${System.nanoTime()}"
    store = createStore(storeName)
  }

  @After
  fun closeStore() {
    store.close()
  }

  private fun record(id: String, embedding: FloatArray, metadata: Map<String, String> = emptyMap()) =
    RetrievalRecord(id, listOf(TextPart(id)), embedding, metadata)

  @Test
  fun emptyStore_hasNoRecords() {
    assertEquals(emptyList<String>(), store.getAllRecordIds())
    assertEquals(emptyList<RetrievalRecord>(), store.get(listOf("missing")))
    assertEquals(emptyList<RetrievalRecord>(), store.search(axisVector(0), 10))
  }

  @Test
  fun upsert_thenGet_returnsEmbeddingAndMetadata() {
    store.upsert(listOf(record("a", axisVector(3), mapOf("k" to "v", "n" to "1"))))

    val fetched = store.get(listOf("a")).single()
    assertEquals("a", fetched.id)
    assertArrayEquals(axisVector(3), fetched.embeddings, 1e-6f)
    assertEquals(mapOf("k" to "v", "n" to "1"), fetched.metadata)
  }

  @Test
  fun get_returnsOnlyExistingIds() {
    store.upsert(listOf(record("a", axisVector(0)), record("b", axisVector(1))))

    assertEquals(setOf("a", "b"), store.get(listOf("a", "missing", "b")).map { it.id }.toSet())
  }

  @Test
  fun upsert_sameId_replacesRecord() {
    assumeTrue("store keeps duplicate ids", upsertReplacesExistingId)
    store.upsert(listOf(record("a", axisVector(0), mapOf("version" to "1"))))
    store.upsert(listOf(record("a", axisVector(1), mapOf("version" to "2"))))

    assertEquals(listOf("a"), store.getAllRecordIds())
    val fetched = store.get(listOf("a")).single()
    assertArrayEquals(axisVector(1), fetched.embeddings, 1e-6f)
    assertEquals(mapOf("version" to "2"), fetched.metadata)
  }

  @Test
  fun search_ordersByDescendingSimilarity() {
    store.upsert(
      listOf(record("x", axisVector(0)), record("y", axisVector(1)), record("z", axisVector(2)))
    )

    val hits = store.search(vectorOf(1f, 3f, 2f), 10)

    assertEquals(listOf("y", "z", "x"), hits.map { it.id })
  }

  @Test
  fun search_returnsEmbeddingAndMetadata() {
    store.upsert(listOf(record("a", axisVector(0), mapOf("k" to "v"))))

    val hit = store.search(axisVector(0), 1).single()
    assertArrayEquals(axisVector(0), hit.embeddings, 1e-6f)
    assertEquals(mapOf("k" to "v"), hit.metadata)
  }

  @Test
  fun search_limitsResultsToTopK() {
    store.upsert((0 until 5).map { record("r$it", axisVector(it)) })

    val hits = store.search(vectorOf(5f, 4f, 3f, 2f, 1f), 2)

    assertEquals(listOf("r0", "r1"), hits.map { it.id })
  }

  @Test
  fun search_withMetadataFilter_onlyReturnsRecordsMatchingAllEntries() {
    store.upsert(
      listOf(
        record("a", axisVector(0), mapOf("parent" to "p1", "kind" to "frame")),
        record("b", axisVector(1), mapOf("parent" to "p1", "kind" to "clip")),
        record("c", axisVector(2), mapOf("parent" to "p2", "kind" to "frame")),
        record("d", axisVector(3)),
      )
    )

    assertEquals(
      setOf("a", "b"),
      store.search(axisVector(2), 10, mapOf("parent" to "p1")).map { it.id }.toSet(),
    )
    assertEquals(
      listOf("a"),
      store.search(axisVector(2), 10, mapOf("parent" to "p1", "kind" to "frame")).map { it.id },
    )
    assertEquals(
      emptyList<String>(),
      store.search(axisVector(2), 10, mapOf("parent" to "nope")).map { it.id },
    )
  }

  @Test
  fun search_withMetadataFilter_appliesTopKAfterFiltering() {
    store.upsert(
      listOf(
        record("best-unfiltered", axisVector(0), mapOf("group" to "other")),
        record("g1", axisVector(1), mapOf("group" to "g")),
        record("g2", axisVector(2), mapOf("group" to "g")),
      )
    )

    val hits = store.search(vectorOf(3f, 2f, 1f), 1, mapOf("group" to "g"))

    assertEquals(listOf("g1"), hits.map { it.id })
  }

  @Test
  fun delete_byIds_removesOnlyThoseRecords() {
    store.upsert(listOf(record("a", axisVector(0)), record("b", axisVector(1))))

    store.delete(listOf("a", "missing"))

    assertEquals(listOf("b"), store.getAllRecordIds())
    assertEquals(listOf("b"), store.search(axisVector(0), 10).map { it.id })
  }

  @Test
  fun delete_byMetadataFilter_removesMatchingRecords() {
    store.upsert(
      listOf(
        record("a", axisVector(0), mapOf("parent" to "p1")),
        record("b", axisVector(1), mapOf("parent" to "p1")),
        record("c", axisVector(2), mapOf("parent" to "p2")),
      )
    )

    store.delete(mapOf("parent" to "p1"))

    assertEquals(listOf("c"), store.getAllRecordIds())
  }

  @Test
  fun deleteAll_removesEverything() {
    store.upsert(listOf(record("a", axisVector(0)), record("b", axisVector(1))))

    store.deleteAll()

    assertEquals(emptyList<String>(), store.getAllRecordIds())
    assertEquals(emptyList<RetrievalRecord>(), store.search(axisVector(0), 10))
  }

  @Test
  fun reopen_keepsRecords() {
    assumeTrue("store is not persistent", persistsAcrossReopen)
    store.upsert(listOf(record("a", axisVector(0), mapOf("k" to "v"))))
    store.close()

    store = createStore(storeName)

    assertEquals(listOf("a"), store.getAllRecordIds())
    val hit = store.search(axisVector(0), 1).single()
    assertEquals(mapOf("k" to "v"), hit.metadata)
    assertTrue(hit.embeddings.isNotEmpty())
  }
}
