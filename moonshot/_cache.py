# Copyright 2017-2023 QuantRocket LLC - All Rights Reserved
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
import six
import inspect
import itertools
import pandas as pd
from quantrocket.db import list_databases

TMP_DIR = os.environ.get("MOONSHOT_CACHE_DIR", "/tmp")

class Cache:
    """
    Pickle-based cache for caching arbitrary objects (typically DataFrames)
    based on a key which can also be an arbitrary object.

    Examples
    --------
    Set and get a DataFrame from the cache based on the prices index and columns,
    in backtests only, and don't use the cache if the file containing the strategy
    code was modified more recently than the DataFrame was cached.

    >>> from moonshot._cache import Cache
    >>>
    >>> class MyStrategy(Moonshot):
    >>>
    >>>     def prices_to_signals(self, prices):
    >>>
    >>>         my_dataframe = None
    >>>
    >>>         if self.is_backtest:
    >>>             # try to load from cache
    >>>             cache_key = [prices.index.tolist(), prices.columns.tolist()]
    >>>             my_dataframe = Cache.get(cache_key, prefix="my_df", unless_file_modified=self)
    >>>
    >>>         if my_dataframe is None:
    >>>             # calculate dataframe here
    >>>             my_dataframe = expensive_calculations()
    >>>             if self.is_backtest:
    >>>                 Cache.set(cache_key, my_dataframe, prefix="my_df")
    """

    @classmethod
    def _get_filepath(cls, key_obj, prefix=None):
        """
        Returns a filepath to use for caching a pickle. The filename contains
        a hex digest of the key_obj, ensuring that the cache won't be used if
        the key_obj changes.
        """
        digest = hashlib.sha224(pickle.dumps(key_obj)).hexdigest()
        filepath = "{tmpdir}/moonshot_{prefix}_{digest}.pkl".format(
            tmpdir=TMP_DIR, prefix=prefix, digest=digest)
        return filepath

    @classmethod
    def get(cls, key_obj, prefix=None, unless_file_modified=None, unless_dbs_modified=None):
        """
        Returns an object from cache, or None if it is not available or
        expired.

        Parameters
        ----------
        key_obj : obj, required
            the object used as the cache key (a hash of the object
            is used, therefore the object must be identical to the
            original object but need not be the original object)

        prefix : str, optional
            the prefix that was used the cache key, if any

        unless_file_modified : str or class or class instance, optional
            don't return cached object if this file (or the file this
            class or class instance is defined in) was modified after
            the object was cached

        unless_dbs_modified : dict, optional
            don't return cached object if any of these dbs were modified
            after the object was cached. Pass a dict of kwargs to pass
            to list_databases, for example:
            {"services":["history"], "codes":["my-db"]}

        Returns
        -------
        obj or None
            the cached object

        Examples
        --------
        See class docstring for typical usage.
        """

        filepath = cls._get_filepath(key_obj, prefix=prefix)
        if not os.path.exists(filepath):
            return None

        cache_last_modified = os.path.getmtime(filepath)

        if unless_file_modified is not None:

            if not isinstance(unless_file_modified, six.string_types):

                if hasattr(unless_file_modified, "__module__"):
                    unless_file_modified = inspect.getmodule(unless_file_modified)
                elif hasattr(unless_file_modified, "__class__"):
                    unless_file_modified = unless_file_modified.__class__

                unless_file_modified = inspect.getfile(unless_file_modified)

            watch_file_last_modified = os.path.getmtime(unless_file_modified)

            if watch_file_last_modified > cache_last_modified:
                return None

        if unless_dbs_modified:
            unless_dbs_modified["detail"] = True
            databases = list_databases(**unless_dbs_modified)
            databases = pd.DataFrame.from_records(
                itertools.chain(databases["sqlite"], databases["postgres"]))
            # databases might be empty if testing with a real-time aggregate
            # database because list_databases doesn't report on aggregate
            # databases, only tick databases. Ideally we should translate the
            # aggregate code to the corresponding tick db code and pass that
            # to list_databases, but that is not implemented.
            if not databases.empty:
                db_last_modified = databases.last_modified.dropna().max()
                if not pd.isnull(db_last_modified):
                    db_last_modified = time.mktime(pd.Timestamp(db_last_modified).timetuple())
                    if db_last_modified > cache_last_modified:
                        return None

        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj

    @classmethod
    def set(cls, key_obj, obj_to_cache, prefix=None):
        """
        Caches an arbitrary object using pickle.

        Parameters
        ----------
        obj_to_cache : object, required
            an arbitrary object to cache using pickle

        key_obj : object, required
            an arbitrary object to use as the cache key (a hash of the object
            will be used as the key)

        prefix : str, optional
            a prefix to use for the cache key (in case the key_obj is used for
            caching multiple objects)

        Returns
        -------
        None

        Examples
        --------
        See class docstring for typical usage.
        """
        filepath = cls._get_filepath(key_obj, prefix=prefix)
        with open(filepath, "wb") as f:
            pickle.dump(obj_to_cache, f)
