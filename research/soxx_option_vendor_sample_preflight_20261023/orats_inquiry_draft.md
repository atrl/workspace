# ORATS 数据询样草稿（未发送）

To: support@orats.com
Subject: SOXX historical near-EOD option chain sample and license questions

Hello ORATS Support,

I am evaluating your Delayed Data API historical strikes for internal research on SOXX options. Before purchasing, could you confirm SOXX coverage and provide a small sample or a per-date availability report for these U.S. trading dates: 2011-01-04, 2012-01-04, 2017-01-03, 2022-06-01, 2024-01-03, and 2026-07-01? Please include explicit zero-row responses for dates without SOXX options.

For each available date, I need original option root and adjusted-contract identifiers, expiration, strike, put/call, 15:46 ET quote timestamp, ordered bid/ask, direct midpoint IV and/or smoothed IV, Delta, and any fields identifying the delivered or revised version. Your near-EOD FAQ says put Delta equals the reported call Delta minus 1; please confirm this convention applies across the requested years. In particular, please confirm whether both the +0.25 Delta call and −0.25 Delta put can be bracketed at expiries around 30 calendar days without filling missing quotes.

Could you also clarify:

1. Whether the $199/month Delayed Data API plan includes `datav2/hist/strikes` for SOXX back to 2011, the applicable historical request limit, and any SOXX-specific exclusions.
2. The first delivery time and revision policy for each historical near-EOD file or API record. Our earliest next-day research decision is 09:29 New York time, so the prior session's data must have been available before then.
3. How SOXX splits, symbol/root changes, and adjusted option contracts are represented across 2011–2026.
4. Whether the license permits an internal, access-controlled NAS backup of the exact data received for reproducible research, with no redistribution.
5. Whether a free SOXX-specific sample is available before subscribing, and the complete price and cancellation terms for one month of historical API access. Please distinguish the $199/month historical Data API from the $599 one-time near-EOD archive and the $99/month forward-only FTP feed.

This is a data qualification request only. We are not placing an order or asking for trading advice.

Thank you.
