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

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.runner.RunWith

/**
 * [VectorStoreBenchmark] for [ObjectBoxVectorStore] on a device, searching at least
 * [ObjectBoxBenchmarkStorage.WIDE_SEARCH_COUNT] candidates.
 * Runs like [SqliteVectorStoreBenchmark].
 */
@RunWith(AndroidJUnit4::class)
class ObjectBoxVectorStoreWideSearchBenchmark : VectorStoreBenchmark() {
  override val dataDir = DeviceBenchmark.dataDir
  override val outputDir = DeviceBenchmark.outputDir
  override val device = DeviceBenchmark.device
  override val persistent = true

  private val storage = ObjectBoxBenchmarkStorage(DeviceBenchmark.storageDir)

  override fun createStore(name: String) =
    storage.create(name, minSearchCount = ObjectBoxBenchmarkStorage.WIDE_SEARCH_COUNT)

  override fun storageBytes(name: String) = storage.bytes(name)

  override fun deleteStorage(name: String) = storage.delete(name)
}
