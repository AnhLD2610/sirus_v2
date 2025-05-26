elif row["type"] == "swap":
                        tgt_price = ((self.tgt.pool.slot0.sqrtPriceX96 / 2 * 96) * 2) / 10 ** (
                                    self.tgt.decimal1 - self.tgt.decimal0)
                        vmp_price = ((self.vampire.pool.slot0.sqrtPriceX96 / 2 * 96) * 2) / 10 ** (
                                    self.vampire.decimal1 - self.vampire.decimal0)
                        amount0 = int(row["amount0"])
                        amount1 = int(row["amount1"])
                        amountSpecified = amount0 if amount0 > 0 else amount1
                        zeroForOne = amount0 > 0
                        sqrtPriceLimitX96 = sqrtPriceLimit0 if zeroForOne else sqrtPriceLimit1
                        # price impact:
                        price_impact_tgt, (amt_out_tgt, _, _, _) = self.tgt.pool.price_impact(zeroForOne, amountSpecified,
                                                                                              sqrtPriceLimitX96)
                        price_impact_vmp, (amt_out_vmp, _, _, _) = self.vampire.pool.price_impact(zeroForOne,
                                                                                                  amountSpecified,
                                                                                                  sqrtPriceLimitX96)
                        output_tgt = self.tgt.pool.swap(self.tgt.accs[0], zeroForOne, amountSpecified, sqrtPriceLimitX96)
                        swap_price = (self.tgt.pool.slot0.sqrtPriceX96 / 2 * 96) * 2
                        vol_tgt = abs(amountSpecified)
                        fee_gain_tgt = abs(amountSpecified) * self.tgt.pool.fee / 1e6
                        if zeroForOne:
                            fee_gain_tgt *= swap_price
                            vol_tgt *= swap_price
                            _vol_t0 += abs(amountSpecified)/10**self.tgt.decimal0
                        else:
                            _vol_t1 += abs(amountSpecified) / 10 ** self.tgt.decimal1

                        step_fee['tgt'] += fee_gain_tgt
                        step_vol['tgt'] += vol_tgt
                        if amt_out_vmp > amt_out_tgt:  # execute swap in tgt and vampire pool
                            output_vmp = self.vampire.pool.swap(self.vampire.accs[0], zeroForOne, amountSpecified,
                                                                sqrtPriceLimitX96)
                            swap_price = (self.vampire.pool.slot0.sqrtPriceX96 / 2 * 96) * 2
                            fee_gain_vmp = amountSpecified * self.vampire.pool.fee / 1e6
                            vol_vmp = abs(amountSpecified)
                            if zeroForOne:  # convert from token0 to token1, using the price after swap
                                fee_gain_vmp = fee_gain_vmp * swap_price
                                vol_vmp = abs(amountSpecified) * swap_price
                            step_fee['vmp'] += fee_gain_vmp
                            step_vol['vmp'] += vol_vmp
                            stolen_swaps += 1
                        has_swap = True
                        swap_summary = StepSummary(step_count=self.step_count,
                                                   index=self._current_index,
                                                   block_number=self._current_block,
                                                   event="swap",
                                                   pool="target" if amt_out_tgt > amt_out_vmp else "vampire",
                                                   amount0=amount0,
                                                   amount1=amount1,
                                                   tgt_price=tgt_price,
                                                   vmp_price=vmp_price,
                                                   stolen_swaps=stolen_swaps,
                                                   amt_out_tgt=amt_out_tgt,
                                                   amt_out_vmp=amt_out_vmp,
                                                   price_impact_tgt=price_impact_tgt,
                                                   price_impact_vmp=price_impact_vmp,
                                                   fee_tgt=fee_gain_tgt,
                                                   fee_vmp=fee_gain_vmp,
                                                   volume_tgt=vol_tgt,
                                                   volume_vmp=vol_vmp,
                                                   fee_tier_vmp=self.vampire.pool.fee,
                                                   liquidity_tgt=self.tgt.pool.liquidity,
                                                   liquidity_vmp=self.vampire.pool.liquidity
                                                   )
                        swap_summary.save(self.log_events)