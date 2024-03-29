/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "timeman.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "search.h"
#include "ucioption.h"

namespace Stockfish {

TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }
TimePoint TimeManagement::elapsed() const { return now() - startTime; }

// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)
void TimeManagement::init(Search::LimitsType& limits,
                          Color               us,
                          int                 ply,
                          const OptionsMap&   options) {
    // If we have no time, no need to initialize TM, except for the start time,
    // which is used by movetime.
    startTime = limits.startTime;
    if (limits.time[us] == 0)
        return;

    TimePoint moveOverhead = TimePoint(MoveOverhead);

    // optScale is a percentage of available time to use for the current move.
    // maxScale is a multiplier applied to optimumTime.
    double optScale, maxScale;

    // Maximum move horizon of 50 moves
    int mtg = 50;

    // if less than one second, gradually reduce mtg
    if (limits.time[us] < 1000 && (double(mtg) / limits.time[us] > 0.05))
    {
        mtg = limits.time[us] * 0.05;
    }

    // if less than one second, gradually reduce mtg
    if (limits.time[us] < 1000 && (double(mtg) / limits.time[us] > 0.05))
    {
        mtg = limits.time[us] * 0.05;
    }

    // Make sure timeLeft is > 0 since we may use it as a divisor
    TimePoint timeLeft = std::max(TimePoint(1), limits.time[us] + limits.inc[us] * (mtg - 1)
                                                  - moveOverhead * (2 + mtg));

    // x basetime (+ z increment)
    // If there is a healthy increment, timeLeft can exceed the actual available
    // game time for the current move, so also cap to a percentage of available game time.
    {
        // Use extra time with larger increments
        double optExtra = limits.inc[us] < 500 ? 1.0 : 1.13;

        // Calculate time constants based on current time left.
        double optConstant =
          std::min(0.00308 + 0.000319 * std::log10(limits.time[us] / 1000.0), 0.00506);
        double maxConstant = std::max(3.39 + 3.01 * std::log10(limits.time[us] / 1000.0), 2.93);

        optScale = std::min(0.0122 + std::pow(ply + 2.95, 0.462) * optConstant,
                            0.213 * limits.time[us] / double(timeLeft))
                 * optExtra;
        maxScale = std::min(6.64, maxConstant + ply / 12.0);
    }

    // Limit the maximum possible time for this move
    optimumTime = TimePoint(optScale * timeLeft);
    maximumTime =
      TimePoint(std::min(0.825 * limits.time[us] - moveOverhead, maxScale * optimumTime)) - 10;

    if (options["Ponder"])
        optimumTime += optimumTime / 4;
}

}  // namespace Stockfish
