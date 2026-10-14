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
 * [VectorStoreBenchmark] for [ObjectBoxVectorStore] on the JVM, with ObjectBox's default search
 * width (as many candidates as requested results).
 * Runs like [InMemoryVectorStoreBenchmark].
 */
class ObjectBoxVectorStoreBenchmark : VectorStoreBenchmark() {
  override val dataDir: File? = System.getProperty("vectorData")?.let(::File)
  override val persistent = true

  private val storage =
    ObjectBoxBenchmarkStorage(File(System.getProperty("java.io.tmpdir"), "objectbox-benchmark"))

  override fun createStore(name: String) = storage.create(name, minSearchCount = 0)

  override fun storageBytes(name: String) = storage.bytes(name)

  override fun deleteStorage(name: String) = storage.delete(name)
}
