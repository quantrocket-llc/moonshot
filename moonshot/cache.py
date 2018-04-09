# Copyright 2018 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hashlib
import pickle
import time
import pandas as pd

TMP_DIR = os.environ.get("MOONSHOT_CACHE_DIR", "/tmp")

class HistoryCache(object):

    @classmethod
    def get_filepath(cls, kwargs):
        """
        Returns a filepath to use for caching a history pickle. The filename
        contains a hex digest of the query kwargs, ensuring that the cache
        won't be used if the kwargs change.
        """
        digest = hashlib.sha224(pickle.dumps(kwargs)).hexdigest()
        filepath = "{tmpdir}/moonshot_history_{digest}.pkl".format(
            tmpdir=TMP_DIR, digest=digest)
        return filepath

    @classmethod
    def load(cls, kwargs, max_cache_age, timezone=None):
        """
        Loads from a pickle if the pickle exists and is newer than max_cache_age.
        """

        filepath = cls.get_filepath(kwargs)
        if not os.path.exists(filepath):
            return None

        last_modified = os.path.getmtime(filepath)
        fileage = time.time() - last_modified

        allowed_age = pd.Timedelta(max_cache_age)

        if fileage > allowed_age.total_seconds():
            return None

        prices = pd.read_pickle(filepath)
        return prices

    @classmethod
    def dump(cls, prices, kwargs):
        """
        Dumps a prices DataFrame to a pickle.
        """
        filepath = cls.get_filepath(kwargs)
        prices.to_pickle(filepath)
