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

import java.io.File

/**
 * Creates, measures and deletes the [ObjectBoxVectorStore] databases of [VectorStoreBenchmark]
 * subclasses, one directory per store name in [root].
 */
class ObjectBoxBenchmarkStorage(private val root: File) {
  fun create(name: String, minSearchCount: Int) =
    ObjectBoxVectorStore(File(root, name), minSearchCount = minSearchCount)

  fun bytes(name: String): Long =
    File(root, name).walk().filter { it.isFile }.sumOf { it.length() }

  fun delete(name: String) {
    File(root, name).deleteRecursively()
  }

  companion object {
    /** Wide-search variant: HNSW searches at least this many candidates. */
    const val WIDE_SEARCH_COUNT = 100
  }
}
