# SOP-01 Digester temperature control

## Normal operating window
The digester runs in the mesophilic range. Target temperature is 37 C. The operating window is 35 C to 39 C (instrument TT_101).
Methanogenic bacteria are sensitive to changes: a variation of more than 1 C per day reduces biogas production and methane content.

## Alarm TAL - Digester temperature low (below 35 C)
1. Check the heating boiler status on the SCADA overview (equipment E_110). A boiler trip is the most common cause.
2. If the boiler tripped, reset it once from the local panel. Do not reset more than once without maintenance approval.
3. Reduce the substrate feeding rate (FT_103) by 30 % until temperature is back above 36 C. Cold substrate makes the drop worse.
4. Check the recirculation pump P_105. Without recirculation the heat exchanger cannot transfer heat.
5. Inform the shift supervisor and record the event in the logbook with start time and minimum temperature.

## Recovery
Raise temperature slowly: maximum 0.5 C per hour. A fast increase stresses the biology as much as a fast drop.
Expect lower biogas flow and CH4 below 58 % for 12 to 24 hours after a temperature upset.
Monitor pH every 2 hours during recovery (see SOP-02).

## Alarm TAH - Digester temperature high (above 39 C)
Stop the heating loop, keep recirculation running and check the TT_101 sensor against the local thermometer TI_101.
