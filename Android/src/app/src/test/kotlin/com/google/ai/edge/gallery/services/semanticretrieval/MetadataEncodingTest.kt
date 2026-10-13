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

import com.google.ai.edge.gallery.services.semanticretrieval.SemanticRetrievalService.Companion.decodeMetadataFromStore
import com.google.ai.edge.gallery.services.semanticretrieval.SemanticRetrievalService.Companion.encodeMetadataForStore
import org.junit.Assert.assertEquals
import org.junit.Test

/** Escaping of metadata keys that collide with SqliteVectorStore's column names. */
class MetadataEncodingTest {
  @Test
  fun encode_prefixesReservedAndAlreadyPrefixedKeys() {
    val encoded =
      encodeMetadataForStore(mapOf("text" to "a", "parent_id" to "b", "_user_x" to "c", "k" to "d"))

    assertEquals(
      mapOf("_user_text" to "a", "_user_parent_id" to "b", "_user__user_x" to "c", "k" to "d"),
      encoded,
    )
  }

  @Test
  fun decode_reversesEncode() {
    val metadata =
      SemanticRetrievalService.SQLITE_RESERVED_METADATA_KEYS.associateWith { "v-$it" } +
        mapOf("_user_x" to "prefixed", "plain" to "value")

    assertEquals(metadata, decodeMetadataFromStore(encodeMetadataForStore(metadata)))
  }

  @Test
  fun decode_prefersEscapedKeyOverStoreColumnOfSameName() {
    // SqliteVectorStore merges its own column values (e.g. "text") into the returned metadata.
    val fromStore = mapOf("text" to "column value", "_user_text" to "user value")

    assertEquals(mapOf("text" to "user value"), decodeMetadataFromStore(fromStore))
  }
}
