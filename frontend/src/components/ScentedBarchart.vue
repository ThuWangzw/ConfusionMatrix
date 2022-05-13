<template>
    <svg :id="widgetId" width="100%" height="100%" ref="svg"></svg>
</template>

<script>
import * as d3 from 'd3';
window.d3 = d3;
import Util from './Util.vue';
import GlobalVar from './GlovalVar.vue';

export default {
    name: 'ScentedBarchart',
    mixins: [Util, GlobalVar],
    props: {
        allSize: {
            type: Array,
            default: undefined,
        },
        title: {
            type: String,
            default: '',
        },
        selectSize: {
            type: Array,
            default: undefined,
        },
    },
    computed: {
        widgetId: function() {
            return 'scented-barchart-svg-'+this.title;
        },
        // allSizeRectG: function() {
        //     return d3.selectAll('.allSizeRect');
        // },
        mainSvg: function() {
            return d3.select('#'+this.widgetId);
        },
    },
    mounted: function() {

    },
    watch: {
        allSize: function() {
            this.drawBarchart();
        },
        selectSize: function() {
            this.render();
        },
    },
    data: function() {
        return {
            globalAttrs: {
                'marginTop': 20, // top margin, in pixels
                'marginRight': 30, // right margin, in pixels
                'marginBottom': 30, // bottom margin, in pixels
                'marginLeft': 30, // left margin, in pixels
                'width': 300, // outer width of chart, in pixels
                'height': 100, // outer height of chart, in pixels
                'insetLeft': 0.5, // inset left edge of bar
                'insetRight': 0.5, // inset right edge of bar:
                'xType': d3.scaleLinear, // type of x-scale
                'yType': d3.scaleLinear, // type of y-scale
            },
            selectSizeRectG: null,
        };
    },
    methods: {
        drawBarchart: function() {
            const that = this;
            // Copyright 2021 Observable, Inc.
            // Released under the ISC license.
            // https://observablehq.com/@d3/histogram
            const drawBarchart = function Histogram(data, {
                label, // convenience alias for xLabel
                normalize, // whether to normalize values to a total of 100%
                yLabel = 'log count', // a label for the y-axis
                yFormat = normalize ? '%' : undefined, // a format specifier string for the y-axis
                color = 'currentColor', // bar fill color
            } = {}) {
                // Compute values.
                // const X = d3.map(data, x);
                // const Y0 = d3.map(data, y);
                // const I = d3.range(X.length);

                // Compute bins.
                // const bins = d3.bin().thresholds(thresholds).value((i) => X[i])(I);
                // const Y = Array.from(bins, (I) => d3.sum(I, (i) => Y0[i]));
                const yRange = [that.globalAttrs['height'] - that.globalAttrs['marginBottom'], that.globalAttrs['marginTop']]; // [bottom, top]
                const xRange = [that.globalAttrs['marginLeft'], that.globalAttrs['width'] - that.globalAttrs['marginRight']]; // [left, right]
                const Y = data;
                if (normalize) {
                    const total = d3.sum(Y);
                    for (let i = 0; i < Y.length; ++i) Y[i] /= total;
                }
                const bins = [];
                for (let i = 0; i < Y.length; ++i) {
                    bins.push({
                        'val': Y[i],
                        'x0': i*0.1,
                        'x1': (i+1)*0.1,
                    });
                }

                // Compute default domains.
                const xDomain = [0, 1];
                const yDomain = [0, Math.log10(d3.max(Y))+1];

                // Construct scales and axes.
                const xScale = that.globalAttrs['xType'](xDomain, xRange);
                const yScale = that.globalAttrs['yType'](yDomain, yRange);
                const xFormat = undefined;
                const xAxis = d3.axisBottom(xScale).ticks(that.globalAttrs['width'] / 80, xFormat).tickSizeOuter(0);
                const yAxis = d3.axisLeft(yScale).ticks(that.globalAttrs['height'] / 40, yFormat);
                yFormat = yScale.tickFormat(100, yFormat);

                const svg = that.mainSvg;

                svg.append('g')
                    .attr('transform', `translate(${that.globalAttrs['marginLeft']},0)`)
                    .call(yAxis)
                    .call((g) => g.select('.domain').remove())
                    .call((g) => g.selectAll('.tick line').clone()
                        .attr('x2', that.globalAttrs['width'] - that.globalAttrs['marginLeft'] - that.globalAttrs['marginRight'])
                        .attr('stroke-opacity', 0.1))
                    .call((g) => g.append('text')
                        .attr('x', -that.globalAttrs['marginLeft'])
                        .attr('y', 10)
                        .attr('fill', 'currentColor')
                        .attr('text-anchor', 'start')
                        .text(yLabel));

                svg.append('g')
                    .attr('fill', color)
                    .selectAll('rect')
                    .data(bins)
                    .join('rect')
                    .attr('x', (d) => xScale(d.x0) + that.globalAttrs['insetLeft'])
                    .attr('width', (d) => Math.max(0, xScale(d.x1) - xScale(d.x0) - that.globalAttrs['insetLeft'] - that.globalAttrs['insetRight']))
                    .attr('y', (d, i) => yScale(Math.log10(Y[i])))
                    .attr('height', (d, i) => yScale(0) - yScale(Math.log10(Y[i])))
                    .attr('class', 'allSizeRect')
                    .append('title')
                    .text((d, i) => [`${d.x0.toFixed(1)} ≤ x < ${d.x1.toFixed(1)}`, yFormat(Math.round(Y[i]))].join('\n'));

                svg.append('g')
                    .attr('transform', `translate(0,${that.globalAttrs['height'] - that.globalAttrs['marginBottom']})`)
                    .call(xAxis)
                    .call((g) => g.append('text')
                        .attr('x', that.globalAttrs['width'] - that.globalAttrs['marginRight'])
                        .attr('y', 27)
                        .attr('fill', 'currentColor')
                        .attr('text-anchor', 'end')
                        .text(label));

                return svg.node();
            };
            drawBarchart(this.allSize, {
                label: this.title,
                color: 'steelblue',
            });
        },
        render: async function() {
            const Y = this.selectSize;
            const bins = [];
            for (let i = 0; i < Y.length; ++i) {
                bins.push({
                    'val': Y[i],
                    'x0': i*0.1,
                    'x1': (i+1)*0.1,
                });
            }
            this.selectSizeRectG = this.mainSvg.selectAll('g.selectSizeRect').data(bins);
            await this.remove();
            await this.update();
            await this.transform();
            await this.create();
        },
        create: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const yRange = [that.globalAttrs['height'] - that.globalAttrs['marginBottom'], that.globalAttrs['marginTop']]; // [bottom, top]
                const yDomain = [0, Math.log10(d3.max(that.allSize))+1];
                const xDomain = [0, 1];
                const xRange = [that.globalAttrs['marginLeft'], that.globalAttrs['width'] - that.globalAttrs['marginRight']]; // [left, right]

                const xScale = that.globalAttrs['xType'](xDomain, xRange);
                const yScale = that.globalAttrs['yType'](yDomain, yRange);

                const selectSizeRectG = that.selectSizeRectG.enter()
                    .append('g')
                    .attr('class', 'selectSizeRect');

                selectSizeRectG.transition()
                    .duration(that.createDuration)
                    .attr('opacity', 1)
                    .on('end', resolve);

                selectSizeRectG.append('rect')
                    .attr('x', (d) => xScale(d.x0) + that.globalAttrs['insetLeft'])
                    .attr('width', (d) => Math.max(0, xScale(d.x1) - xScale(d.x0) - that.globalAttrs['insetLeft'] - that.globalAttrs['insetRight']))
                    .attr('y', (d, i) => yScale(Math.log10(Math.max(1, d.val))))
                    .attr('height', (d, i) => yScale(0) - yScale(Math.log10(Math.max(1, d.val))))
                    .attr('fill', 'orange');

                if ((that.selectSizeRectG.enter().size() === 0)) {
                    resolve();
                }
            });
        },
        update: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                const yRange = [that.globalAttrs['height'] - that.globalAttrs['marginBottom'], that.globalAttrs['marginTop']]; // [bottom, top]
                const yDomain = [0, Math.log10(d3.max(that.allSize))+1];
                const yScale = that.globalAttrs['yType'](yDomain, yRange);
                that.selectSizeRectG.each(function(d, i) {
                    // eslint-disable-next-line no-invalid-this
                    d3.select(this).select('rect')
                        .transition()
                        .duration(that.updateDuration)
                        .attr('y', yScale(Math.log10(Math.max(1, d.val))))
                        .attr('height', yScale(0) - yScale(Math.log10(Math.max(1, d.val))))
                        .on('end', resolve);
                });

                if ((that.selectSizeRectG.size() === 0)) {
                    resolve();
                }
            });
        },
        remove: async function() {
            const that = this;
            return new Promise((resolve, reject) => {
                that.selectSizeRectG.exit()
                    .transition()
                    .duration(that.removeDuration)
                    .attr('opacity', 0)
                    .remove()
                    .on('end', resolve);

                if ((that.selectSizeRectG.exit().size() === 0)) {
                    resolve();
                }
            });
        },
        transform: async function() {
        },
    },
};
</script>
